"""Measured-performance store — calibrates theoretical estimates.

The modeling tools produce *first-order theoretical* (roofline) numbers.
Real systems deviate because of kernel / framework maturity, scheduling,
and quantization-kernel quality — sometimes enough that a newer GPU
underperforms an older one (e.g. B200 below H100 on immature software).

This module persists REAL measured benchmark results in a shared,
cross-session JSONL store and exposes two tools:

* ``record_measurement`` — save a measured result (from ``benchmark_serving``
  or user-reported). It also computes and stores the *corresponding*
  theoretical roofline (via the serving simulator at the same operating
  point) plus the efficiency factor (measured ÷ theory), so every record
  documents both reality and theory side by side.
* ``lookup_measurements`` — fetch recorded results so the agent can report
  a reality-adjusted estimate alongside the theoretical one.

Store location: ``$AGENT_MEASUREMENTS_DIR`` (default ``measurements/``),
resolved at import — shared across all sessions launched from the same
directory, like the memory store.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..base import tool

STORE_DIR = Path(os.environ.get("AGENT_MEASUREMENTS_DIR", "measurements")).resolve()
STORE_FILE = STORE_DIR / "measurements.jsonl"

_MAX_RETURNED = 50


def _ensure() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    STORE_FILE.touch(exist_ok=True)


def _load_all() -> list[dict]:
    _ensure()
    out: list[dict] = []
    for line in STORE_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _load_workload(workload_file: str) -> dict | None:
    """Parse a WorkloadProfile YAML inside the workspace. Returns the dict
    or None on failure (so callers can error gracefully without raising)."""
    try:
        from ...workspace import resolve
        import yaml
        return yaml.safe_load(resolve(workload_file).read_text()) or {}
    except Exception:
        return None


def _theoretical(wf: dict, gpu: str, n_gpus: int) -> dict:
    """Theoretical (roofline) numbers via the serving simulator, run at the
    *same* workload the benchmark consumed (the parsed WorkloadProfile dict).
    Returns metric fields on success or ``{"note": why}`` when it can't /
    shouldn't be computed. Never raises — recording must not depend on the
    model being computable.

    Because the simulator and the benchmark share the same workload knobs
    (model/rate/in/out/num_requests/range_ratio), theory and measurement are
    apples-to-apples by construction. Under saturation this matters: mean
    TTFT/TPOT grow with run length until the queue reaches steady state, so a
    short theoretical run would understate the roofline."""
    # Lazy imports: keep the modeling stack (numpy) off the import path until a
    # record actually needs it.
    from ..modeling.configs.model_specs import PRESET_MODELS
    from ..modeling.configs.hw_specs import PRESET_GPUS

    model = wf.get("model", "")
    request_rate_raw = wf.get("request_rate", 0)

    if model not in PRESET_MODELS or gpu not in PRESET_GPUS:
        missing = "model" if model not in PRESET_MODELS else "gpu"
        return {"note": f"not computed: {missing} is not a PRESET name "
                        "(use a PRESET_MODELS/PRESET_GPUS key to get a baseline)"}
    if n_gpus != 1:
        return {"note": f"not computed: deployment spans {n_gpus} GPUs — the "
                        "modeling tools assume a single GPU and don't model "
                        "TP/PP/DP scaling, so a single-GPU roofline wouldn't "
                        "correspond"}
    try:
        rate_f = float(request_rate_raw)
    except (TypeError, ValueError):
        rate_f = 0.0
    if not (rate_f > 0 and rate_f != float("inf")):
        return {"note": f"not computed: request_rate={request_rate_raw!r} is "
                        "not a finite positive number (the simulator is "
                        "Poisson-arrival-driven; closed-loop runs aren't "
                        "directly comparable to its open-loop baseline)"}
    try:
        from ..modeling.serving import run_simulation, summarize_run
        # Use the workload's own num_requests; floor at 200 for stable
        # saturated-regime percentiles when the workload didn't set one.
        n_req = int(wf.get("num_requests", 0)) or max(int(rate_f * 10), 200)
        result = run_simulation(
            model_name=model, gpu_name=gpu, request_rate=rate_f,
            input_len=int(wf.get("input_len", 0)),
            output_len=int(wf.get("output_len", 0)),
            n_requests=n_req,
            max_batched_tokens=int(wf.get("max_num_batched_tokens", 0)) or 8192,
            jitter=float(wf.get("range_ratio", 0.0)),
        )
        s = summarize_run(result)
        if s is None:
            return {"note": "not computed: simulation produced no finished requests"}
        out = {
            "basis": "single-GPU roofline (simulate_serving, Poisson arrivals)",
            "served_rate_rps": round(s.served_rate, 3),
            "ttft_ms": round(s.ttft_ms.mean, 3),
            "tpot_ms": round(s.tpot_ms.mean, 3),
            "mean_in_flight": round(s.mean_in_flight, 2),
        }
        if s.saturated:
            out["saturated"] = True
            out["saturation_reason"] = s.saturation_reason
        return out
    except Exception as e:  # never let theory break the record
        return {"note": f"not computed: {type(e).__name__}: {e}"}


def _efficiency(rec: dict, theo: dict) -> dict:
    """Fraction of theoretical ideal achieved (1.0 = matches theory, <1 = worse).
    Stored for TTFT and TPOT only (theory/measured — theory is the floor).
    Throughput isn't stored: at sub-saturation it just mirrors the offered
    load, so the ratio isn't a capability signal."""
    eff: dict = {}
    for key in ("ttft_ms", "tpot_ms"):
        m, t = rec.get(key), theo.get(key)
        if m and t:
            eff[key.replace("_ms", "")] = round(t / m, 3)
    return eff


def _parallelism_str(r: dict) -> str:
    """Compact parallelism descriptor; omits dims left at their default."""
    parts = [f"tp={r.get('tensor_parallel', 1)}"]
    if r.get("pipeline_parallel", 1) != 1:
        parts.append(f"pp={r['pipeline_parallel']}")
    if r.get("data_parallel", 1) != 1:
        parts.append(f"dp={r['data_parallel']}")
    if r.get("expert_parallel"):
        parts.append("ep")
    return " ".join(parts)


def _theory_str(r: dict) -> str:
    """Render the stored theoretical baseline + efficiency, if present."""
    t = r.get("theoretical")
    if not t:
        return ""
    if "ttft_ms" not in t and "tpot_ms" not in t:  # only a 'note' (not computed)
        return f"\n      theory: {t.get('note', 'n/a')}"
    seg = []
    if t.get("served_rate_rps"):
        seg.append(f"rate={t['served_rate_rps']:g} req/s")
    if t.get("ttft_ms"):
        seg.append(f"TTFT={t['ttft_ms']:g}ms")
    if t.get("tpot_ms"):
        seg.append(f"TPOT={t['tpot_ms']:g}ms")
    line = f"\n      theory ({t.get('basis', 'roofline')}): " + ", ".join(seg)
    if t.get("saturated"):
        line += "  [SATURATED in theory]"
    eff = r.get("efficiency", {})
    effseg = []
    if "ttft" in eff:
        effseg.append(f"TTFT {eff['ttft']:.0%}")
    if "tpot" in eff:
        effseg.append(f"TPOT {eff['tpot']:.0%}")
    if effseg:
        line += "  | efficiency: " + ", ".join(effseg) + " of ideal"
    return line


def _fmt(r: dict) -> str:
    metrics = []
    if r.get("ttft_ms"):
        metrics.append(f"TTFT={r['ttft_ms']:g}ms")
    if r.get("tpot_ms"):
        metrics.append(f"TPOT={r['tpot_ms']:g}ms")
    metric_str = ", ".join(metrics) or "(no metrics)"
    rate = r.get("request_rate")
    rate_str = f"rate={rate} req/s, " if rate else ""
    head = (f"{r.get('model','?')} on {r.get('gpu','?')}"
            f" | {rate_str}c={r.get('concurrency','?')} (peak in-flight) "
            f"in={r.get('input_len','?')} out={r.get('output_len','?')} "
            f"{_parallelism_str(r)}")
    tail = f" | source={r.get('source','?')}"
    notes = f" — {r['notes']}" if r.get("notes") else ""
    ts = r.get("ts", "")
    return f"[{ts}] {head} | measured: {metric_str}{tail}{notes}{_theory_str(r)}"


@tool(
    "Record a REAL, measured benchmark result for a workload+GPU so future "
    "estimates can be calibrated against it. Theoretical models are "
    "first-order and can be off due to kernel/framework maturity — call this "
    "whenever you learn real measured numbers (`benchmark_serving` calls it "
    "automatically; otherwise user-reported). Pass the SAME `workload_file` "
    "the benchmark consumed — the workload context (model, request_rate, "
    "input_len, output_len, num_requests, range_ratio, …) is read from there "
    "and the matching theoretical roofline is computed via the simulator on "
    "the same workload, so theory and measurement are by-construction "
    "apples-to-apples. Stored metrics are TTFT and TPOT — the primitive "
    "serving-performance numbers; throughput isn't stored because at "
    "sub-saturation it just mirrors `request_rate × output_len`.",
    workload_file="Workspace-relative path to the WorkloadProfile YAML the "
                  "benchmark ran (e.g. 'stages/01_workload.yaml').",
    gpu="GPU name (prefer a PRESET_GPUS key).",
    concurrency="Observed peak in-flight requests during the measurement.",
    ttft_ms="Measured time-to-first-token in ms; 0 if unknown.",
    tpot_ms="Measured time-per-output-token in ms; 0 if unknown.",
    tensor_parallel="Tensor-parallel size (TP) of the deployment (default 1).",
    pipeline_parallel="Pipeline-parallel size (PP) of the deployment (default 1).",
    data_parallel="Data-parallel size (DP) of the deployment, e.g. replicas "
                  "(default 1). Total GPUs = TP×PP×DP.",
    expert_parallel="Whether expert parallelism (EP) was used for an MoE model "
                    "(default False).",
    source="Where the number came from, e.g. 'user-reported' or 'vllm bench'.",
    notes="Free-text context: framework + version, and why it may differ "
          "from theory (e.g. 'immature B200 kernels, ~0.7x of H100').",
)
def record_measurement(
    workload_file: str,
    gpu: str,
    concurrency: int,
    ttft_ms: float = 0.0,
    tpot_ms: float = 0.0,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    data_parallel: int = 1,
    expert_parallel: bool = False,
    source: str = "user-reported",
    notes: str = "",
) -> str:
    wf = _load_workload(workload_file)
    if wf is None:
        return (f"ERROR: could not load workload_file {workload_file!r} "
                f"(must be a workspace-relative path to a WorkloadProfile YAML)")

    _ensure()
    n_gpus = tensor_parallel * pipeline_parallel * data_parallel
    request_rate = str(wf.get("request_rate", ""))
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workload_file": workload_file,
        "model": wf.get("model", ""),
        "gpu": gpu,
        "request_rate": request_rate,
        "concurrency": concurrency,   # observed peak in-flight at this rate
        "input_len": wf.get("input_len"),
        "output_len": wf.get("output_len"),
        "num_requests": wf.get("num_requests"),
        "tensor_parallel": tensor_parallel,
        "pipeline_parallel": pipeline_parallel,
        "data_parallel": data_parallel,
        "expert_parallel": expert_parallel,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "source": source,
        "notes": notes,
    }
    # Run the simulator at the same workload as the benchmark — apples-to-apples
    # by construction. Stores measured + theoretical + efficiency in one record.
    theo = _theoretical(wf, gpu, n_gpus)
    rec["theoretical"] = theo
    eff = _efficiency(rec, theo)
    if eff:
        rec["efficiency"] = eff

    with STORE_FILE.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    rate_bit = f"rate={request_rate} req/s, " if request_rate else ""
    status = (f"recorded measurement: {rec['model']} on {gpu} "
              f"({rate_bit}c={concurrency}, in={rec['input_len']}, "
              f"out={rec['output_len']}, {_parallelism_str(rec)})")
    if eff:
        bits = []
        if "ttft" in eff:
            bits.append(f"TTFT {eff['ttft']:.0%}")
        if "tpot" in eff:
            bits.append(f"TPOT {eff['tpot']:.0%} of roofline")
        if bits:
            status += " | efficiency vs theory: " + ", ".join(bits)
    elif theo.get("note"):
        status += f" | theory {theo['note']}"
    return status


@tool(
    "Look up previously recorded REAL benchmark measurements to calibrate a "
    "theoretical estimate. Filter by model and/or gpu (leave a field blank "
    "to match any; substring match). Call this after a theoretical estimate "
    "to check whether measured performance is known for this model+GPU, "
    "then report the theoretical number AND a reality-adjusted estimate.",
    model="Filter to this model (blank = any).",
    gpu="Filter to this GPU (blank = any).",
)
def lookup_measurements(model: str = "", gpu: str = "") -> str:
    recs = _load_all()

    def matches(r: dict) -> bool:
        if model and model.lower() not in str(r.get("model", "")).lower():
            return False
        if gpu and gpu.lower() not in str(r.get("gpu", "")).lower():
            return False
        return True

    hits = [r for r in recs if matches(r)]
    if not hits:
        scope = " + ".join(x for x in (model, gpu) if x) or "(any)"
        return (f"No recorded measurements for {scope}. The estimate is "
                f"purely theoretical — say so, and offer to record real "
                f"numbers if the user has them.")
    hits = hits[-_MAX_RETURNED:]
    lines = [f"{len(hits)} measurement(s) found:"]
    lines += [f"  - {_fmt(r)}" for r in hits]
    lines.append("Each record carries the measured numbers AND the "
                 "corresponding theoretical roofline, with the efficiency "
                 "factor (fraction of ideal achieved). Present both, and use "
                 "the efficiency factor to reality-adjust new estimates for "
                 "this model+GPU.")
    return "\n".join(lines)
