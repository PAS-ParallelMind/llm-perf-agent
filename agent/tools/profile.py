"""Profiling tools for CUDA-oriented workspace tasks."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

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

    action_hints = []
    if "uncoalesced_global_access" in bottlenecks:
        action_hints.extend([
            "improve global memory coalescing",
            "reorganize memory accesses with tiling",
            "use shared memory as a staging tile when it preserves semantics",
        ])
    if "memory_bound" in bottlenecks:
        action_hints.extend([
            "reduce unnecessary global/L2 traffic",
            "prioritize memory access pattern changes over arithmetic changes",
        ])
    if "occupancy_gap" in bottlenecks and "memory_bound" not in bottlenecks:
        action_hints.append("inspect block size, register pressure, and launch configuration")

    if not action_hints:
        action_hints.append("compare source with profiler hints before making targeted changes")

    return {
        "bottlenecks": bottlenecks or ["unknown"],
        "summary": _diagnosis_sentence(metrics, findings, bottlenecks),
        "action_hints": action_hints,
    }


def _diagnosis_sentence(metrics: dict, findings: list[dict], bottlenecks: list[str]) -> str:
    if "uncoalesced_global_access" in bottlenecks:
        finding = next(f for f in findings if f.get("type") == "uncoalesced_global_access")
        return (
            "Profiler indicates uncoalesced global memory accesses with "
            f"{finding['excessive_sector_ratio_pct']}% excessive sectors; focus on coalescing and data layout."
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


def _profile_command(run_command: str, profiler: str) -> tuple[str, str]:
    if profiler == "none":
        return run_command, "none"
    if profiler == "nvprof":
        return f"nvprof {run_command}", "nvprof"
    if profiler == "nsys":
        return f"nsys profile --stats=true --force-overwrite=true -o profile_report {run_command}", "nsys"
    if profiler == "ncu":
        return f"ncu --target-processes all {run_command}", "ncu"

    if shutil.which("nvprof"):
        return f"nvprof {run_command}", "nvprof"
    if shutil.which("nsys"):
        return f"nsys profile --stats=true --force-overwrite=true -o profile_report {run_command}", "nsys"
    if shutil.which("ncu"):
        return f"ncu --target-processes all {run_command}", "ncu"
    return run_command, "none"


@tool(
    "Build and profile a CUDA benchmark command inside the workspace. "
    "Use this before and after edits to measure whether optimization helped. "
    "It optionally runs a build command, then runs the target command one or "
    "more times with nvprof/nsys/ncu when available, falling back to plain "
    "execution. Returns JSON with exit codes, parsed time: ...ms values, "
    "checksums, and output tails.",
    workdir="Directory relative to workspace where commands should run, e.g. 'BankRedux'",
    run_command="Command to execute from workdir, e.g. './sum_cuda 1024000' or 'sh test.sh'",
    build_command="Optional build command to run first, e.g. 'make'",
    profiler="auto, none, nvprof, nsys, or ncu. Default auto",
    repeats="Number of profiling repeats. Default 1",
    timeout="Timeout per command in seconds. Default 300",
    detail="summary, compact, or full. Default summary; use compact/full only when more raw output is needed",
)
def cuda_profile(
    workdir: str,
    run_command: str,
    build_command: str = "",
    profiler: str = "auto",
    repeats: int = 1,
    timeout: int = 300,
    detail: str = "summary",
) -> str:
    cwd = resolve(workdir) if workdir else get_root()
    if not cwd.is_dir():
        return f"ERROR: workdir is not a directory: {workdir!r}"
    if repeats < 1:
        repeats = 1
    if detail not in {"summary", "compact", "full"}:
        detail = "summary"

    result: dict = {"workdir": str(cwd.relative_to(get_root()))}
    combined = ""

    if build_command:
        build = _run(build_command, cwd, timeout)
        result["build"] = _compact_build(build, detail)
        combined += "\n" + build.get("output", "")
        if build.get("exit_code") != 0:
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
    result["ok"] = all(r.get("exit_code") == 0 for r in runs)
    return json.dumps(result, indent=2)


@tool(
    "Run a guided CUDA profiling pass from broad to narrow. This is designed "
    "for LLM optimization: it runs an optional build, a direct correctness/run "
    "command, Nsight Compute basic metrics, then optional detailed rule and "
    "source-counter passes. It returns compact JSON with metrics, profiler "
    "findings, diagnosis, and action-family hints instead of raw profiler logs.",
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
                "action_hints": ["inspect build output", "repair source or Makefile before optimizing"],
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
            "action_hints": [
                "inspect direct_run output",
                "use one of available_executables if the binary name is wrong",
                "restore correctness before optimizing performance",
            ],
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
            "action_hints": ["use cuda_profile or direct timing", "install or load ncu for guided bottleneck diagnosis"],
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
    result["diagnosis"] = _diagnose(metrics, findings)
    result["parsed"] = _parse(combined)
    result["ok"] = basic.get("exit_code") == 0 and (not detailed_run or detailed_run.get("exit_code") == 0)
    return json.dumps(result, indent=2)
