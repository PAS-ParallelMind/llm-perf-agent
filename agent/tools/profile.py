"""Profiling tools for CUDA-oriented workspace tasks."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
import csv
import io
import shlex
from statistics import mean, stdev

from ..workspace import get_root, resolve
from .base import tool

MAX_OUT = 30_000
TIME_RE = re.compile(r"time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
CHECKSUM_RE = re.compile(r"checksum:\s*([^\s,]+)", re.IGNORECASE)
PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
NUMBER_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
PROFILE_LINE_RE = re.compile(
    r"(GPU activities|API calls|CUDA Kernel|CUDA API|memcpy|cudaMemcpy|"
    r"cudaMalloc|cudaFree|LaunchKernel|Kernel Name|Time\(%\)|Duration|"
    r"Name\s*$|^[\s0-9.]+(?:us|ms|s)\s+)",
    re.IGNORECASE,
)
NCU_METRICS = {
    "duration": "Duration",
    "memory_throughput_pct": "Memory Throughput",
    "dram_throughput_pct": "DRAM Throughput",
    "l1tex_throughput_pct": "L1/TEX Cache Throughput",
    "l2_throughput_pct": "L2 Cache Throughput",
    "compute_throughput_pct": "Compute (SM) Throughput",
    "achieved_occupancy_pct": "Achieved Occupancy",
    "theoretical_occupancy_pct": "Theoretical Occupancy",
    "l2_hit_rate_pct": "L2 Hit Rate",
}


def _truncate(text: str, limit: int = MAX_OUT) -> str:
    return text if len(text) <= limit else text[-limit:] + "\n... [truncated from front]"


def _cuda_env() -> dict:
    env = os.environ.copy()
    for prefix in ("/usr/local/cuda", "/opt/cuda"):
        nvcc = Path(prefix) / "bin" / "nvcc"
        if nvcc.exists():
            env["CUDA_HOME"] = prefix
            env["PATH"] = str(nvcc.parent) + os.pathsep + env.get("PATH", "")
            lib64 = Path(prefix) / "lib64"
            if lib64.exists():
                env["LD_LIBRARY_PATH"] = str(lib64) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            break
    return env


def _run(command: str, cwd: Path, timeout: int) -> dict:
    env = _cuda_env()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return {
            "command": command,
            "exit_code": proc.returncode,
            "output": _truncate(out),
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = e.stderr if isinstance(e.stderr, str) else ""
        return {
            "command": command,
            "exit_code": None,
            "timeout": True,
            "output": _truncate(stdout + stderr),
        }


def _parse(output: str) -> dict:
    times = [float(x) for x in TIME_RE.findall(output)]
    checksums = CHECKSUM_RE.findall(output)
    return {
        "times_ms": times,
        "best_time_ms": min(times) if times else None,
        "last_time_ms": times[-1] if times else None,
        "checksums": checksums,
    }


def _available_executables(cwd: Path) -> list[str]:
    names = []
    for path in sorted(cwd.iterdir()):
        if path.is_file() and os.access(path, os.X_OK):
            names.append("./" + path.name)
    return names[:20]


def _profile_summary(output: str, max_lines: int = 80) -> list[str]:
    lines = []
    for line in output.splitlines():
        stripped = line.rstrip()
        if PROFILE_LINE_RE.search(stripped):
            lines.append(stripped)
        if len(lines) >= max_lines:
            break
    return lines


def _metric_value(line: str) -> float | None:
    pct = PCT_RE.search(line)
    if pct:
        return float(pct.group(1))
    nums = NUMBER_RE.findall(line)
    return float(nums[-1]) if nums else None


def _extract_ncu_metrics(output: str) -> dict:
    metrics: dict = {}
    for line in output.splitlines():
        for key, label in NCU_METRICS.items():
            if label in line and key not in metrics:
                value = _metric_value(line)
                if value is not None:
                    metrics[key] = value
                    if key == "duration":
                        if " ms" in line or "msecond" in line:
                            metrics["duration_unit"] = "ms"
                        elif " us" in line or "usecond" in line:
                            metrics["duration_unit"] = "us"
                        elif " ns" in line or "nsecond" in line:
                            metrics["duration_unit"] = "ns"
    return metrics


def _extract_rule_findings(output: str) -> list[dict]:
    findings = []
    for match in re.finditer(
        r"Est\.?\s+Speedup:\s*([0-9.]+)%.*?uncoalesced global accesses.*?"
        r"total of\s*([0-9,]+)\s*excessive sectors\s*"
        r"\(([0-9.]+)%\s*of the total\s*([0-9,]+)\s*sectors\)",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        findings.append({
            "type": "uncoalesced_global_access",
            "estimated_speedup_pct": float(match.group(1)),
            "excessive_sectors": int(match.group(2).replace(",", "")),
            "excessive_sector_ratio_pct": float(match.group(3)),
            "total_sectors": int(match.group(4).replace(",", "")),
        })
    for match in re.finditer(r"Est\.?\s+Speedup:\s*([0-9.]+)%\s*(.{0,280})", output, flags=re.IGNORECASE):
        text = " ".join(match.group(2).split())
        if text and not any(f.get("message") == text for f in findings):
            findings.append({
                "type": "profiler_speedup_hint",
                "estimated_speedup_pct": float(match.group(1)),
                "message": text,
            })
    return findings[:8]


def _diagnose(metrics: dict, findings: list[dict]) -> dict:
    compute = metrics.get("compute_throughput_pct")
    memory = metrics.get("memory_throughput_pct")
    l2 = metrics.get("l2_throughput_pct")
    achieved = metrics.get("achieved_occupancy_pct")
    theoretical = metrics.get("theoretical_occupancy_pct")

    bottlenecks = []
    if compute is not None and (memory is not None or l2 is not None):
        mem_pressure = max(v for v in (memory, l2) if v is not None)
        if compute < 25 and mem_pressure > 65:
            bottlenecks.append("memory_bound")
    if any(f.get("type") == "uncoalesced_global_access" for f in findings):
        bottlenecks.append("uncoalesced_global_access")
    if achieved is not None and theoretical is not None and theoretical - achieved > 20:
        bottlenecks.append("occupancy_gap")

    return {
        "bottlenecks": bottlenecks or ["unknown"],
        "summary": _diagnosis_sentence(metrics, findings, bottlenecks),
    }


def _diagnosis_sentence(metrics: dict, findings: list[dict], bottlenecks: list[str]) -> str:
    if "uncoalesced_global_access" in bottlenecks:
        finding = next(f for f in findings if f.get("type") == "uncoalesced_global_access")
        return (
            "Profiler indicates uncoalesced global memory accesses with "
            f"{finding['excessive_sector_ratio_pct']}% excessive sectors."
        )
    if "memory_bound" in bottlenecks:
        return (
            "Kernel appears memory-bound: memory/L2 throughput is high while compute throughput is low."
        )
    if "occupancy_gap" in bottlenecks:
        return "Profiler shows a significant achieved-vs-theoretical occupancy gap."
    return "No dominant bottleneck was automatically identified from the summarized profiler output."


def _compact_run(run_result: dict, detail: str) -> dict:
    output = run_result.get("output", "")
    compact = {
        "command": run_result.get("command"),
        "exit_code": run_result.get("exit_code"),
        "timeout": run_result.get("timeout", False),
        "parsed": _parse(output),
        "profile_summary": _profile_summary(output),
    }
    if detail == "compact" or compact["exit_code"] not in (0, None):
        compact["output_tail"] = _truncate(output, 6000)
    elif detail == "full":
        compact["output"] = output
    return compact


def _compact_build(build_result: dict, detail: str) -> dict:
    output = build_result.get("output", "")
    compact = {
        "command": build_result.get("command"),
        "exit_code": build_result.get("exit_code"),
        "timeout": build_result.get("timeout", False),
    }
    if detail == "full":
        compact["output"] = output
    elif detail == "compact" or compact["exit_code"] != 0:
        compact["output_tail"] = _truncate(output, 6000)
    return compact


def _csv_rows_with_header(output: str, header_prefix: str) -> list[dict]:
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            return list(csv.DictReader(io.StringIO("\n".join(lines[i:]))))
    return []


def _nsys_trace_summary(output: str, max_events: int = 40) -> dict:
    rows = _csv_rows_with_header(output, "Start (ns),")
    events = []
    for row in rows:
        try:
            start = int(row.get("Start (ns)", "") or 0)
            duration = int(row.get("Duration (ns)", "") or 0)
        except ValueError:
            continue
        name = row.get("Name", "")
        category = "kernel"
        if "memcpy" in name.lower():
            category = "memcpy"
        events.append({
            "start_ns": start,
            "end_ns": start + duration,
            "duration_ns": duration,
            "category": category,
            "stream": row.get("Strm") or None,
            "name": name,
            "bytes_mb": _float_or_none(row.get("Bytes (MB)")),
            "src_mem": row.get("SrcMemKd") or None,
            "dst_mem": row.get("DstMemKd") or None,
        })

    if not events:
        return {"events": [], "overlap": {}}

    first = min(e["start_ns"] for e in events)
    compact_events = []
    for event in sorted(events, key=lambda e: e["start_ns"])[:max_events]:
        compact_events.append({
            "start_us": round((event["start_ns"] - first) / 1000.0, 3),
            "duration_us": round(event["duration_ns"] / 1000.0, 3),
            "category": event["category"],
            "stream": event["stream"],
            "name": event["name"],
            "bytes_mb": event["bytes_mb"],
            "src_mem": event["src_mem"],
            "dst_mem": event["dst_mem"],
        })

    operation_sequence = []
    for event in sorted(events, key=lambda e: e["start_ns"]):
        label = _timeline_event_label(event)
        if not operation_sequence or operation_sequence[-1] != label:
            operation_sequence.append(label)
        if len(operation_sequence) >= 24:
            break

    memcpy_events = [e for e in events if e["category"] == "memcpy"]
    kernel_events = [e for e in events if e["category"] == "kernel"]
    overlap_ns = 0
    for memcpy in memcpy_events:
        for kernel in kernel_events:
            overlap_ns += max(0, min(memcpy["end_ns"], kernel["end_ns"]) - max(memcpy["start_ns"], kernel["start_ns"]))

    kernel_ns = sum(e["duration_ns"] for e in kernel_events)
    memcpy_ns = sum(e["duration_ns"] for e in memcpy_events)
    span_ns = max(e["end_ns"] for e in events) - first
    return {
        "events": compact_events,
        "operation_sequence": operation_sequence,
        "overlap": {
            "memcpy_events": len(memcpy_events),
            "kernel_events": len(kernel_events),
            "total_memcpy_us": round(memcpy_ns / 1000.0, 3),
            "total_kernel_us": round(kernel_ns / 1000.0, 3),
            "timeline_span_us": round(span_ns / 1000.0, 3),
            "memcpy_kernel_overlap_us": round(overlap_ns / 1000.0, 3),
            "kernel_overlap_ratio": round(overlap_ns / kernel_ns, 4) if kernel_ns else None,
        },
    }


def _timeline_event_label(event: dict) -> str:
    if event.get("category") == "kernel":
        return "kernel"
    name = event.get("name", "")
    if "Host-to-Device" in name:
        return "H2D"
    if "Device-to-Host" in name:
        return "D2H"
    if "Device-to-Device" in name:
        return "D2D"
    return "memcpy"


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _runtime_summary(parsed: dict) -> dict:
    samples = parsed.get("times_ms") or []
    numeric = [float(x) for x in samples if isinstance(x, (int, float))]
    nonzero = [x for x in numeric if x > 0]
    chosen = nonzero or numeric
    return {
        "samples": numeric,
        "mean": round(mean(chosen), 4) if chosen else None,
        "stdev": round(stdev(chosen), 4) if len(chosen) > 1 else 0.0 if chosen else None,
    }


def _nsys_gpu_sum(output: str) -> dict:
    rows = _csv_rows_with_header(output, "Time (%),")
    hot_kernels = []
    h2d = 0.0
    d2h = 0.0
    memcpy = 0.0
    for row in rows:
        time_pct = _float_or_none(row.get("Time (%)")) or 0.0
        operation = row.get("Operation", "")
        category = row.get("Category", "")
        instances = int(float(row.get("Instances", "0") or 0))
        if category == "CUDA_KERNEL":
            hot_kernels.append({
                "name": operation.strip('"'),
                "time_pct": round(time_pct, 2),
                "instances": instances,
            })
        elif category == "MEMORY_OPER":
            memcpy += time_pct
            if "Host-to-Device" in operation:
                h2d += time_pct
            elif "Device-to-Host" in operation:
                d2h += time_pct
    return {
        "hot_kernels": sorted(hot_kernels, key=lambda x: x["time_pct"], reverse=True)[:5],
        "transfer_pct": {
            "h2d": round(h2d, 2),
            "d2h": round(d2h, 2),
            "total": round(memcpy, 2),
        },
    }


def _profile_analysis(
    *,
    parsed: dict,
    metrics: dict | None = None,
    findings: list[dict] | None = None,
    timeline: dict | None = None,
    gpu_sum_output: str = "",
) -> dict:
    metrics = metrics or {}
    findings = findings or []
    gpu_sum = _nsys_gpu_sum(gpu_sum_output) if gpu_sum_output else {"hot_kernels": [], "transfer_pct": {}}
    transfer = gpu_sum.get("transfer_pct") or {}
    hot_kernels = gpu_sum.get("hot_kernels") or []

    sm = metrics.get("compute_throughput_pct")
    dram = metrics.get("dram_throughput_pct")
    memory = metrics.get("memory_throughput_pct")
    occupancy = metrics.get("achieved_occupancy_pct")
    memory_bound = bool(
        (dram is not None and sm is not None and dram > sm * 2 and dram > 50)
        or (memory is not None and sm is not None and memory > sm * 2 and memory > 50)
    )

    bottleneck_signals = []
    data_quality_warnings = []
    if memory_bound:
        bottleneck_signals.append("DRAM/memory throughput dominates SM throughput")
    if transfer.get("total", 0) >= 20:
        bottleneck_signals.append(
            f"transfer overhead is material (H2D {transfer.get('h2d', 0)}%, D2H {transfer.get('d2h', 0)}%)"
        )
    if hot_kernels:
        top = hot_kernels[0]
        bottleneck_signals.append(
            f"single dominant kernel ({top['name']}, {top['time_pct']}% of GPU kernel time)"
        )

    overlap = (timeline or {}).get("overlap") or {}
    if overlap:
        bottleneck_signals.append(
            "memcpy/kernel overlap: "
            f"{overlap.get('memcpy_kernel_overlap_us')} us "
            f"(kernel overlap ratio {overlap.get('kernel_overlap_ratio')})"
        )
        if overlap.get("memcpy_kernel_overlap_us") == 0 and transfer.get("total", 0) >= 20:
            bottleneck_signals.append("CUDA timeline shows transfers and kernels are effectively serialized")

    for finding in findings[:3]:
        msg = finding.get("message") or finding.get("type")
        if msg:
            bottleneck_signals.append(str(msg))

    runtime = _runtime_summary(parsed)
    samples = runtime.get("samples") or []
    if any(x == 0 for x in samples):
        data_quality_warnings.append("runtime output contains 0.00 ms samples; treat speedups from these samples as low confidence")
    if runtime.get("mean") and runtime.get("stdev") and runtime["stdev"] / runtime["mean"] > 0.2:
        data_quality_warnings.append("runtime samples have high variance; compare medians across full validation sizes")

    next_measurements = []
    if overlap:
        next_measurements.append("rerun cuda_profile with timeline=auto after stream/copy changes and compare overlap fields")
    if transfer.get("total", 0) >= 20:
        next_measurements.append("compare transfer_pct and runtime across full validation sizes")
    if metrics:
        next_measurements.append("rerun cuda_profile after kernel changes and compare ncu aggregate fields")
    if not next_measurements:
        next_measurements.append("compare before/after runtime on the full validation command")

    host_memory_kinds = sorted({
        str(event.get(key))
        for event in (timeline or {}).get("events", [])
        for key in ("src_mem", "dst_mem")
        if event.get(key) and event.get(key) not in {"Device"}
    })

    profile_facts = {
        "runtime_value": runtime,
        "hot_kernels": hot_kernels,
        "transfer_pct": transfer or None,
        "occupancy_pct": occupancy,
        "sm_utilization": "low" if sm is not None and sm < 30 else "moderate_or_high" if sm is not None else None,
        "memory_bound": memory_bound,
        "host_memory_kinds": host_memory_kinds or None,
        "ncu_aggregate": {
            "sm_throughput_pct": sm,
            "dram_throughput_pct": dram,
            "memory_throughput_pct": memory,
            "achieved_occupancy_pct": occupancy,
            "theoretical_occupancy_pct": metrics.get("theoretical_occupancy_pct"),
            "l2_hit_rate_pct": metrics.get("l2_hit_rate_pct"),
        },
        "timeline_overlap": overlap or None,
        "operation_sequence": (timeline or {}).get("operation_sequence") or None,
    }
    return {
        "profile_facts": profile_facts,
        "profile_interpretation": {
            "bottleneck_signals": bottleneck_signals,
            "data_quality_warnings": data_quality_warnings,
            "confidence": _profile_confidence(runtime, metrics, timeline),
        },
        "next_measurements": next_measurements,
    }


def _profile_confidence(runtime: dict, metrics: dict, timeline: dict | None) -> str:
    samples = runtime.get("samples") or []
    if not samples:
        return "low"
    if any(x == 0 for x in samples):
        return "low"
    if runtime.get("mean") and runtime.get("stdev") and runtime["stdev"] / runtime["mean"] > 0.2:
        return "low"
    if metrics or timeline:
        return "medium"
    return "low"


def _profile_command(run_command: str, profiler: str) -> tuple[str, str]:
    if profiler == "none":
        return run_command, "none"
    if profiler == "nvprof":
        return f"nvprof {run_command}", "nvprof"
    if profiler == "nsys":
        return f"nsys profile --stats=true --force-overwrite=true -o profile_report {run_command}", "nsys"
    if profiler == "ncu":
        return f"ncu --target-processes all {run_command}", "ncu"

    if shutil.which("ncu"):
        return f"ncu --target-processes all {run_command}", "ncu"
    if shutil.which("nsys"):
        return f"nsys profile --stats=true --force-overwrite=true -o profile_report {run_command}", "nsys"
    if shutil.which("nvprof"):
        return f"nvprof {run_command}", "nvprof"
    return run_command, "none"


def _metrics_memory_bound(metrics: dict) -> bool:
    sm = metrics.get("compute_throughput_pct")
    dram = metrics.get("dram_throughput_pct")
    memory = metrics.get("memory_throughput_pct")
    return bool(
        (dram is not None and sm is not None and dram > sm * 2 and dram > 50)
        or (memory is not None and sm is not None and memory > sm * 2 and memory > 50)
    )


def _should_collect_timeline(timeline: str, metrics: dict) -> bool:
    if timeline == "on":
        return True
    if timeline == "off":
        return False
    return _metrics_memory_bound(metrics)


def _collect_nsys_timeline(
    *,
    cwd: Path,
    run_command: str,
    timeout: int,
    max_events: int,
    detail: str,
) -> dict:
    result: dict = {"nsys_available": shutil.which("nsys") is not None}
    combined = ""
    if not result["nsys_available"]:
        result["ok"] = False
        result["diagnosis"] = {
            "summary": "Nsight Systems is unavailable; CUDA timeline overlap could not be collected.",
        }
        return {"result": result, "combined": combined, "timeline": {}, "gpu_sum_output": ""}

    with tempfile.TemporaryDirectory(prefix="pm-nsys-") as tmp:
        report_base = Path(tmp) / "timeline"
        quoted_base = shlex.quote(str(report_base))
        profile_cmd = (
            "nsys profile --trace=cuda --cuda-event-trace=false "
            f"--force-overwrite=true --stats=false -o {quoted_base} {run_command}"
        )
        profile = _run(profile_cmd, cwd, timeout)
        result["nsys_profile"] = _compact_run(profile, detail)
        combined += "\n" + profile.get("output", "")
        report = report_base.with_suffix(".nsys-rep")
        if profile.get("exit_code") != 0 or not report.exists():
            result["ok"] = False
            return {"result": result, "combined": combined, "timeline": {}, "gpu_sum_output": ""}

        stats_cmd = (
            "nsys stats --force-export=true --report cuda_gpu_trace "
            f"--format csv {shlex.quote(str(report))}"
        )
        trace = _run(stats_cmd, cwd, timeout)
        combined += "\n" + trace.get("output", "")
        if detail == "compact":
            result["cuda_gpu_trace"] = _compact_run(trace, "compact")

        sum_cmd = (
            "nsys stats --force-export=true --report cuda_gpu_sum "
            f"--format csv {shlex.quote(str(report))}"
        )
        gpu_sum = _run(sum_cmd, cwd, timeout)
        combined += "\n" + gpu_sum.get("output", "")
        if detail == "compact":
            result["cuda_gpu_summary"] = _compact_run(gpu_sum, "compact")

    timeline_summary = _nsys_trace_summary(trace.get("output", ""), max_events=max_events)
    result["timeline_overlap"] = timeline_summary.get("overlap")
    result["operation_sequence"] = timeline_summary.get("operation_sequence")
    if detail == "compact":
        result["timeline_events"] = timeline_summary.get("events")
    result["ok"] = trace.get("exit_code") == 0
    return {
        "result": result,
        "combined": combined,
        "timeline": timeline_summary,
        "gpu_sum_output": gpu_sum.get("output", ""),
    }


@tool(
    "Build and profile a CUDA benchmark command inside the workspace. "
    "Use this before and after edits to measure whether optimization helped. "
    "It optionally runs a build command, then runs the target command one or "
    "more times with ncu when available, falling back to plain execution. "
    "With timeline=auto/on it also collects Nsight Systems CUDA timeline facts "
    "such as transfer percentage, memory kinds, operation sequence, and "
    "memcpy-vs-kernel overlap. Returns JSON with exit codes, parsed time: ...ms "
    "values, checksums, and profile_information facts/light interpretation.",
    workdir="Directory relative to workspace where commands should run, e.g. 'BankRedux'",
    run_command="Command to execute from workdir, e.g. './sum_cuda 1024000' or 'sh test.sh'",
    build_command="Optional build command to run first, e.g. 'make'",
    profiler="auto, none, ncu, nsys, or nvprof. Default auto prefers local ncu",
    repeats="Number of profiling repeats. Default 1",
    timeout="Timeout per command in seconds. Default 300",
    detail="summary, compact, or full. Default summary; use compact/full only when more raw output is needed",
    timeline="auto, off, or on. Default auto collects nsys timeline when ncu facts indicate it is useful",
    max_events="Maximum nsys timeline events to return only in detail=compact/full when timeline is collected. Default 12",
)
def cuda_profile(
    workdir: str,
    run_command: str,
    build_command: str = "",
    profiler: str = "auto",
    repeats: int = 1,
    timeout: int = 300,
    detail: str = "summary",
    timeline: str = "auto",
    max_events: int = 12,
) -> str:
    cwd = resolve(workdir) if workdir else get_root()
    if not cwd.is_dir():
        return f"ERROR: workdir is not a directory: {workdir!r}"
    if repeats < 1:
        repeats = 1
    if detail not in {"summary", "compact", "full"}:
        detail = "summary"
    if timeline not in {"auto", "off", "on"}:
        timeline = "auto"
    if max_events < 1:
        max_events = 12

    result: dict = {"workdir": str(cwd.relative_to(get_root()))}
    combined = ""
    metrics: dict = {}
    findings: list[dict] = []
    timeline_summary: dict = {}
    gpu_sum_output = ""

    if build_command:
        build = _run(build_command, cwd, timeout)
        result["build"] = _compact_build(build, detail)
        combined += "\n" + build.get("output", "")
        if build.get("exit_code") != 0:
            result["ok"] = False
            result["parsed"] = _parse(combined)
            result["profile_summary"] = _profile_summary(combined)
            return json.dumps(result, indent=2)

    direct = _run(run_command, cwd, timeout)
    result["direct_run"] = _compact_run(direct, detail)
    combined += "\n" + direct.get("output", "")
    if direct.get("exit_code") != 0:
        result["ok"] = False
        result["parsed"] = _parse(combined)
        result["profile_summary"] = _profile_summary(combined)
        return json.dumps(result, indent=2)

    profiled_command, profiler_used = _profile_command(run_command, profiler)
    result["profiler"] = profiler_used
    runs = []
    for _ in range(repeats):
        run_result = _run(profiled_command, cwd, timeout)
        runs.append(_compact_run(run_result, detail))
        combined += "\n" + run_result.get("output", "")
    result["runs"] = runs
    result["parsed"] = _parse(combined)
    if profiler_used == "ncu":
        metrics = _extract_ncu_metrics(combined)
        findings = _extract_rule_findings(combined)
        result["metrics"] = metrics
        result["profiler_rule_findings"] = findings
        result["diagnosis"] = _diagnose(metrics, findings)
    if _should_collect_timeline(timeline, metrics):
        timeline_detail = "compact" if detail in {"compact", "full"} else "summary"
        timeline_result = _collect_nsys_timeline(
            cwd=cwd,
            run_command=run_command,
            timeout=timeout,
            max_events=max_events,
            detail=timeline_detail,
        )
        result["integrated_timeline"] = timeline_result["result"]
        combined += timeline_result["combined"]
        timeline_summary = timeline_result["timeline"]
        gpu_sum_output = timeline_result["gpu_sum_output"]
        result["parsed"] = _parse(combined)
    if profiler_used == "ncu" or timeline_summary:
        result["profile_information"] = _profile_analysis(
            parsed=result["parsed"],
            metrics=metrics,
            findings=findings,
            timeline=timeline_summary,
            gpu_sum_output=gpu_sum_output,
        )
    result["ok"] = all(r.get("exit_code") == 0 for r in runs)
    return json.dumps(result, indent=2)


@tool(
    "Profile CUDA timeline with Nsight Systems. Use this when optimizing "
    "host-device transfers, streams, or overlap. It returns CUDA memcpy/kernel "
    "profile_information facts/light interpretation, compact timeline events, "
    "stream IDs, pageable/pinned memory labels when available, and computed "
    "memcpy-vs-kernel overlap.",
    workdir="Directory relative to workspace where commands should run, e.g. 'HDOverlap'",
    run_command="Command to execute from workdir, e.g. './axpy_cuda 1024000'",
    build_command="Optional build command to run first, e.g. 'make'",
    timeout="Timeout per command in seconds. Default 300",
    max_events="Maximum timeline events to return. Default 40",
    detail="summary or compact. compact includes nsys output tails",
)
def cuda_timeline_profile(
    workdir: str,
    run_command: str,
    build_command: str = "",
    timeout: int = 300,
    max_events: int = 40,
    detail: str = "summary",
) -> str:
    cwd = resolve(workdir) if workdir else get_root()
    if not cwd.is_dir():
        return f"ERROR: workdir is not a directory: {workdir!r}"
    if detail not in {"summary", "compact"}:
        detail = "summary"
    if max_events < 1:
        max_events = 40

    result: dict = {"workdir": str(cwd.relative_to(get_root()))}
    combined = ""

    if build_command:
        build = _run(build_command, cwd, timeout)
        result["build"] = _compact_build(build, detail)
        combined += "\n" + build.get("output", "")
        if build.get("exit_code") != 0:
            result["ok"] = False
            result["parsed"] = _parse(combined)
            return json.dumps(result, indent=2)

    direct = _run(run_command, cwd, timeout)
    result["direct_run"] = _compact_run(direct, detail)
    combined += "\n" + direct.get("output", "")
    if direct.get("exit_code") != 0:
        result["ok"] = False
        result["parsed"] = _parse(combined)
        return json.dumps(result, indent=2)

    if not shutil.which("nsys"):
        result["ok"] = False
        result["nsys_available"] = False
        result["parsed"] = _parse(combined)
        result["diagnosis"] = {
            "summary": "Nsight Systems is unavailable; CUDA timeline overlap could not be collected.",
        }
        return json.dumps(result, indent=2)

    with tempfile.TemporaryDirectory(prefix="pm-nsys-") as tmp:
        report_base = Path(tmp) / "timeline"
        quoted_base = shlex.quote(str(report_base))
        profile_cmd = (
            "nsys profile --trace=cuda --cuda-event-trace=false "
            f"--force-overwrite=true --stats=false -o {quoted_base} {run_command}"
        )
        profile = _run(profile_cmd, cwd, timeout)
        result["nsys_profile"] = _compact_run(profile, detail)
        combined += "\n" + profile.get("output", "")
        report = report_base.with_suffix(".nsys-rep")
        if profile.get("exit_code") != 0 or not report.exists():
            result["ok"] = False
            result["parsed"] = _parse(combined)
            return json.dumps(result, indent=2)

        stats_cmd = (
            "nsys stats --force-export=true --report cuda_gpu_trace "
            f"--format csv {shlex.quote(str(report))}"
        )
        trace = _run(stats_cmd, cwd, timeout)
        combined += "\n" + trace.get("output", "")
        if detail == "compact":
            result["cuda_gpu_trace"] = _compact_run(trace, "compact")

        sum_cmd = (
            "nsys stats --force-export=true --report cuda_gpu_sum "
            f"--format csv {shlex.quote(str(report))}"
        )
        gpu_sum = _run(sum_cmd, cwd, timeout)
        combined += "\n" + gpu_sum.get("output", "")
        if detail == "compact":
            result["cuda_gpu_summary"] = _compact_run(gpu_sum, "compact")

    timeline = _nsys_trace_summary(trace.get("output", ""), max_events=max_events)
    result["parsed"] = _parse(combined)
    result["profile_information"] = _profile_analysis(
        parsed=result["parsed"],
        timeline=timeline,
        gpu_sum_output=gpu_sum.get("output", ""),
    )
    result["timeline_overlap"] = timeline.get("overlap")
    result["timeline_events"] = timeline.get("events")
    result["operation_sequence"] = timeline.get("operation_sequence")
    result["ok"] = trace.get("exit_code") == 0
    return json.dumps(result, indent=2)


@tool(
    "Run a guided CUDA profiling pass from broad to narrow. This is designed "
    "for LLM optimization: it runs an optional build, a direct correctness/run "
    "command, Nsight Compute basic metrics, then optional detailed rule and "
    "source-counter passes. It returns compact JSON with metrics, profiler "
    "findings, and profiler-derived profile_information instead of raw profiler logs.",
    workdir="Directory relative to workspace where commands should run, e.g. 'HDOverlap'",
    run_command="Command to profile from workdir, e.g. './axpy_cuda 1024000'",
    build_command="Optional build command to run first, e.g. 'make'",
    launch_skip="Kernel launches to skip for ncu, e.g. warmups. Default 0",
    launch_count="Kernel launches to profile for ncu. Default 1",
    detailed="Whether to run ncu --set detailed after basic. Default true",
    source_counters="Whether to also run ncu --section SourceCounters. Default false because it can be slow",
    timeout="Timeout per command in seconds. Default 600",
    detail="summary or compact. compact includes profiler output tails",
)
def cuda_guided_profile(
    workdir: str,
    run_command: str,
    build_command: str = "",
    launch_skip: int = 0,
    launch_count: int = 1,
    detailed: bool = True,
    source_counters: bool = False,
    timeout: int = 600,
    detail: str = "summary",
) -> str:
    cwd = resolve(workdir) if workdir else get_root()
    if not cwd.is_dir():
        return f"ERROR: workdir is not a directory: {workdir!r}"
    if detail not in {"summary", "compact"}:
        detail = "summary"

    result: dict = {
        "workdir": str(cwd.relative_to(get_root())),
        "profiling_strategy": [
            "direct run: validate output and parse program-reported time/checksum",
            "ncu basic: classify compute vs memory vs occupancy",
            "ncu detailed: extract profiler rule findings",
            "SourceCounters: optional source-line attribution",
        ],
    }

    if build_command:
        build = _run(build_command, cwd, timeout)
        result["build"] = _compact_build(build, detail)
        if build.get("exit_code") != 0:
            result["ok"] = False
            result["diagnosis"] = {
                "bottlenecks": ["build_failed"],
                "summary": "Build failed before profiling; fix compile errors first.",
            }
            return json.dumps(result, indent=2)

    direct = _run(run_command, cwd, timeout)
    result["direct_run"] = _compact_run(direct, detail)
    if direct.get("exit_code") != 0:
        executables = _available_executables(cwd)
        result["ok"] = False
        if executables:
            result["available_executables"] = executables
        result["diagnosis"] = {
            "bottlenecks": ["run_failed"],
            "summary": "Program failed before Nsight Compute profiling; fix correctness/runtime errors first.",
        }
        return json.dumps(result, indent=2)

    if not shutil.which("ncu"):
        parsed = _parse(direct.get("output", ""))
        result["ok"] = True
        result["ncu_available"] = False
        result["metrics"] = {}
        result["profiler_rule_findings"] = []
        result["diagnosis"] = {
            "bottlenecks": ["ncu_unavailable"],
            "summary": "Nsight Compute is unavailable; only program-reported timing/checksum was collected.",
        }
        result["parsed"] = parsed
        return json.dumps(result, indent=2)

    ncu_base = f"ncu --target-processes all --launch-skip {launch_skip} --launch-count {launch_count}"
    basic = _run(f"{ncu_base} --set basic {run_command}", cwd, timeout)
    detailed_run = None
    source_run = None
    combined = direct.get("output", "") + "\n" + basic.get("output", "")

    if detailed:
        detailed_run = _run(f"{ncu_base} --set detailed {run_command}", cwd, timeout)
        combined += "\n" + detailed_run.get("output", "")
    if source_counters:
        source_run = _run(f"{ncu_base} --section SourceCounters {run_command}", cwd, timeout)
        combined += "\n" + source_run.get("output", "")

    metrics = _extract_ncu_metrics(combined)
    findings = _extract_rule_findings(combined)
    result["ncu_available"] = True
    result["ncu"] = {
        "basic": _compact_run(basic, detail),
        "detailed": _compact_run(detailed_run, detail) if detailed_run else None,
        "source_counters": _compact_run(source_run, detail) if source_run else None,
    }
    result["metrics"] = metrics
    result["profiler_rule_findings"] = findings
    result["profile_information"] = _profile_analysis(
        parsed=result["parsed"],
        metrics=metrics,
        findings=findings,
    )
    result["diagnosis"] = _diagnose(metrics, findings)
    result["parsed"] = _parse(combined)
    result["ok"] = basic.get("exit_code") == 0 and (not detailed_run or detailed_run.get("exit_code") == 0)
    return json.dumps(result, indent=2)
