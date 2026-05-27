"""Continuous-batching serving simulator with Poisson-arrival workload.

The workload is described by an arrival *rate* (req/s, Poisson) plus
per-request (input_len, output_len). Concurrency is then a *result* — it
emerges from the arrival rate, request size, and serving capacity:

* ``max_concurrent_requests`` is a server-policy cap (vLLM ``--max-num-seqs``).
* The KV cache budget caps concurrency further: a request is admitted
  only if its worst-case future KV (prompt+gen tokens, summed across
  layers with SWA caps) fits alongside currently-running requests.

When the input rate exceeds steady-state capacity the report flags the
run as **saturated** — TTFT diverges in that regime so the averages
become meaningless; we surface that explicitly rather than hiding it.

Exposed as the ``simulate_serving`` tool; also runnable as a CLI for
deeper exploration (microbenchmark file support, custom seed, etc.).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass

from ..base import tool
from .configs.hw_specs import PRESET_GPUS
from .configs.model_specs import PRESET_MODELS, ModelConfig, get_quantization_bytes
from .latency import (
    OperationLatency,
    OpBreakdown,
    Request,
    forward_pass_latency,
    total_tokens_in_batch,
)
from .memory import weights_vram_gib
from .report import ReportBuilder


# Fraction of GPU memory reserved for activations / CUDA graphs / framework
# overhead. Matches what vLLM's ``--gpu-memory-utilization 0.85`` defaults to.
GPU_MEMORY_OVERHEAD = 0.15

# Discard the first WARMUP_FRACTION of completed requests when computing
# tail-latency percentiles — they finish before the system reaches steady
# state, so their TTFTs are artificially low.
WARMUP_FRACTION = 0.10

# After all requests finish, decide saturation by comparing served rate
# to requested rate. Below this fraction we flag the run as saturated.
SATURATION_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# KV budget accounting
# ---------------------------------------------------------------------------

def _kv_bytes_per_token_summed(model: ModelConfig, context_tokens: int) -> int:
    """Worst-case KV bytes for one request that grows to ``context_tokens``.

    Sums per-layer cost respecting sliding-window caps (SWA layers hold at
    most ``sliding_window`` tokens regardless of context length).
    """
    kv_byte = get_quantization_bytes(model.kv_cache_dtype)
    bytes_per_token_per_layer = 2 * model.n_kv_heads * model.head_dim * kv_byte
    tokens_summed = sum(
        model.effective_kv_tokens(layer_idx, context_tokens)
        for layer_idx in range(model.n_layers)
    )
    return bytes_per_token_per_layer * tokens_summed


def compute_kv_budget_bytes(model: ModelConfig, gpu, n_gpus: int) -> int:
    """KV-cache budget in bytes after weights + overhead are deducted."""
    total_bytes = int(gpu.mem_capacity * n_gpus)
    weights_bytes = int(weights_vram_gib(model) * (1024 ** 3))
    overhead_bytes = int(total_bytes * GPU_MEMORY_OVERHEAD)
    return max(0, total_bytes - weights_bytes - overhead_bytes)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class ContinuousBatchingScheduler:
    """vLLM-style continuous-batching scheduler.

    Each :meth:`schedule_step` call assigns ``tokens_this_step`` to every
    running request and may admit waiting-queue requests, subject to:

    * ``max_batched_tokens`` — per-step token budget for the forward pass.
    * ``max_concurrent_requests`` — hard cap on in-flight requests.
    * ``kv_budget_bytes``       — KV-cache budget for the GPU(s).
    """

    max_batched_tokens: int
    max_concurrent_requests: int
    kv_budget_bytes: int
    model: ModelConfig

    def _request_worst_case_kv_bytes(self, r: Request) -> int:
        return _kv_bytes_per_token_summed(self.model, r.prompt_tokens + r.gen_tokens)

    def schedule_step(
        self,
        waiting_queue: list[Request],
        running_queue: list[Request],
        kv_in_use_bytes: int,
    ) -> int:
        """Schedule one forward pass. Returns updated ``kv_in_use_bytes``."""
        # 1) reset per-step token assignment.
        for r in running_queue:
            r.tokens_this_step = 0

        # 2) decode tokens for running requests that finished prefill.
        for r in running_queue:
            if r.prompt_tokens <= r.kv_tokens:
                r.tokens_this_step = 1

        # 3) continue prefill for running requests still in their prompt,
        #    bounded by the remaining batched-token budget.
        for r in running_queue:
            if r.prompt_tokens > r.kv_tokens:
                budget_left = self.max_batched_tokens - total_tokens_in_batch(running_queue)
                r.tokens_this_step = min(r.prompt_tokens - r.kv_tokens, budget_left)

        # 4) admit waiting requests subject to all caps.
        while (
            waiting_queue
            and len(running_queue) < self.max_concurrent_requests
            and total_tokens_in_batch(running_queue) < self.max_batched_tokens
        ):
            r = waiting_queue[0]
            need = self._request_worst_case_kv_bytes(r)
            if kv_in_use_bytes + need > self.kv_budget_bytes:
                break  # KV-bound; cannot admit anyone else this step
            budget_left = self.max_batched_tokens - total_tokens_in_batch(running_queue)
            r.tokens_this_step = min(r.prompt_tokens, budget_left)
            waiting_queue.pop(0)
            running_queue.append(r)
            kv_in_use_bytes += need

        return kv_in_use_bytes


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Everything the report needs from one simulation run."""
    finished: list[Request]
    sim_breakdown: OpBreakdown
    n_forward_passes: int
    total_batch_tokens: int
    wall_time_s: float
    requested_rate: float
    n_requests: int

    # System instrumentation
    kv_budget_bytes: int
    peak_kv_use_bytes: int
    peak_in_flight: int
    mean_in_flight: float
    admitted_kv_blocked: int    # times a request was held back by KV budget


def _poisson_arrivals(rng: random.Random, n: int, rate: float) -> list[float]:
    """Cumulative arrival times under a Poisson process with mean rate ``rate``."""
    times: list[float] = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(rate)
        times.append(t)
    return times


def _build_request_pool(
    rng: random.Random,
    n_requests: int,
    input_len: int,
    output_len: int,
    request_rate: float,
    jitter: float,
) -> list[Request]:
    arrivals = _poisson_arrivals(rng, n_requests, request_rate)
    pool: list[Request] = []
    for i, arrival in enumerate(arrivals):
        prompt = max(1, int(rng.uniform(input_len * (1 - jitter), input_len * (1 + jitter))))
        gen = max(1, int(rng.uniform(output_len * (1 - jitter), output_len * (1 + jitter))))
        pool.append(Request(
            id=i, arrival_s=arrival,
            prompt_tokens=prompt, gen_tokens=gen,
        ))
    return pool


def _request_is_done(r: Request) -> bool:
    """A request finishes when its KV cache holds prompt+gen-1 tokens.

    The last generated token is produced but not cached (no further decode
    will read it), matching vLLM-style accounting.
    """
    return r.kv_tokens == r.prompt_tokens + r.gen_tokens - 1


def run_simulation(
    model_name: str,
    gpu_name: str,
    request_rate: float,
    input_len: int,
    output_len: int,
    n_requests: int,
    max_batched_tokens: int,
    max_concurrent_requests: int = 1024,
    n_gpus: int = 1,
    jitter: float = 0.1,
    microbench: dict | None = None,
    seed: int = 0,
    progress_cb=None,
) -> SimulationResult:
    """Run the simulation. Wall clock advances by each step's roofline time."""
    rng = random.Random(seed)
    model = PRESET_MODELS[model_name]
    gpu = PRESET_GPUS[gpu_name]
    kv_budget = compute_kv_budget_bytes(model, gpu, n_gpus)
    scheduler = ContinuousBatchingScheduler(
        max_batched_tokens=max_batched_tokens,
        max_concurrent_requests=max_concurrent_requests,
        kv_budget_bytes=kv_budget,
        model=model,
    )

    arrival_pool = _build_request_pool(rng, n_requests, input_len, output_len, request_rate, jitter)
    waiting_queue: list[Request] = []
    running_queue: list[Request] = []
    finished: list[Request] = []

    sim_breakdown = OpBreakdown()
    expert_cache: dict = {}     # scoped to this simulation
    t_now = 0.0
    n_forward_passes = 0
    total_batch_tokens = 0
    kv_in_use = 0

    peak_kv_use = 0
    peak_in_flight = 0
    in_flight_time_weighted = 0.0
    admitted_kv_blocked = 0

    def _release_kv(r: Request) -> None:
        nonlocal kv_in_use
        kv_in_use = max(0, kv_in_use - _kv_bytes_per_token_summed(
            model, r.prompt_tokens + r.gen_tokens
        ))

    pool_idx = 0
    while pool_idx < len(arrival_pool) or waiting_queue or running_queue:
        # Pull arrivals up to t_now into the waiting queue.
        while pool_idx < len(arrival_pool) and arrival_pool[pool_idx].arrival_s <= t_now:
            waiting_queue.append(arrival_pool[pool_idx])
            pool_idx += 1

        # If running is empty and waiting is empty, jump the clock to the
        # next arrival (no work to do until then).
        if not running_queue and not waiting_queue and pool_idx < len(arrival_pool):
            t_now = arrival_pool[pool_idx].arrival_s
            continue

        # Try to admit before this step's forward pass.
        kv_before_admit = kv_in_use
        kv_in_use = scheduler.schedule_step(waiting_queue, running_queue, kv_in_use)
        if waiting_queue and kv_in_use == kv_before_admit and len(running_queue) > 0:
            # Couldn't admit because of KV. Record once per scheduler step.
            admitted_kv_blocked += 1

        if not running_queue:
            # Still nothing to run (shouldn't happen here given the early
            # check above, but be safe).
            if pool_idx < len(arrival_pool):
                t_now = arrival_pool[pool_idx].arrival_s
            continue

        # Record start_s the first time a request runs.
        for r in running_queue:
            if r.start_s == 0.0 and r.kv_tokens == 0:
                r.start_s = t_now

        # Forward pass.
        step_breakdown = forward_pass_latency(running_queue, microbench, gpu, model, expert_cache)
        sim_breakdown.accumulate(step_breakdown)
        step_duration = step_breakdown.total().roofline_s
        n_forward_passes += 1
        total_batch_tokens += total_tokens_in_batch(running_queue)

        # Time-weighted in-flight tracking.
        in_flight = len(running_queue) + len(waiting_queue)
        in_flight_time_weighted += in_flight * step_duration
        peak_in_flight = max(peak_in_flight, in_flight)
        peak_kv_use = max(peak_kv_use, kv_in_use)

        t_now += step_duration

        # Update each request after the pass.
        still_running: list[Request] = []
        for r in running_queue:
            r.kv_tokens += r.tokens_this_step
            prefill_complete = r.kv_tokens >= r.prompt_tokens
            if prefill_complete and r.first_token_s == 0.0:
                r.first_token_s = t_now
            if _request_is_done(r):
                r.finish_s = t_now
                _release_kv(r)
                finished.append(r)
                if progress_cb is not None:
                    progress_cb(1)
            else:
                still_running.append(r)
        running_queue[:] = still_running

    mean_in_flight = (in_flight_time_weighted / t_now) if t_now > 0 else 0.0

    return SimulationResult(
        finished=finished,
        sim_breakdown=sim_breakdown,
        n_forward_passes=n_forward_passes,
        total_batch_tokens=total_batch_tokens,
        wall_time_s=t_now,
        requested_rate=request_rate,
        n_requests=n_requests,
        kv_budget_bytes=kv_budget,
        peak_kv_use_bytes=peak_kv_use,
        peak_in_flight=peak_in_flight,
        mean_in_flight=mean_in_flight,
        admitted_kv_blocked=admitted_kv_blocked,
    )


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------

@dataclass
class Percentiles:
    mean: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def of(cls, values: list[float]) -> "Percentiles":
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0)
        sorted_vals = sorted(values)
        return cls(
            mean=statistics.fmean(sorted_vals),
            p50=_quantile(sorted_vals, 0.50),
            p95=_quantile(sorted_vals, 0.95),
            p99=_quantile(sorted_vals, 0.99),
        )


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation quantile on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


@dataclass
class RunSummary:
    # Workload
    requested_rate: float
    served_rate: float
    n_finished: int

    # Latency percentiles (ms / s)
    ttft_ms: Percentiles
    tpot_ms: Percentiles
    e2e_s: Percentiles
    wait_ms: Percentiles

    # System
    wall_time_s: float
    mean_in_flight: float
    peak_in_flight: int
    kv_budget_gib: float
    peak_kv_use_gib: float
    admitted_kv_blocked: int

    # Saturation
    saturated: bool
    saturation_reason: str

    # Per-forward-pass averages
    n_forward_passes: int
    avg_batch_tokens: float
    avg_op_per_step: OpBreakdown


def _avg_op(op: OperationLatency, n: int) -> OperationLatency:
    return OperationLatency(op.compute_s / n, op.memory_s / n, op.roofline_s / n)


def _avg_breakdown(breakdown: OpBreakdown, n: int) -> OpBreakdown:
    avg = OpBreakdown()
    for name, op in breakdown.iter_ops():
        setattr(avg, name, _avg_op(op, n))
    return avg


def _saturation_reason(result: SimulationResult, served_rate: float) -> tuple[bool, str]:
    if result.requested_rate <= 0:
        return False, ""
    if served_rate >= SATURATION_THRESHOLD * result.requested_rate:
        return False, ""
    kv_pressure = result.peak_kv_use_bytes / max(result.kv_budget_bytes, 1)
    if kv_pressure >= 0.95 and result.admitted_kv_blocked > 0:
        return True, (
            f"KV-cache bound (peak {kv_pressure * 100:.0f}% of budget; "
            f"{result.admitted_kv_blocked} admission stalls). "
            "Reduce max context, add VRAM, or lower request_rate."
        )
    return True, (
        "Compute-bound: the forward-pass throughput cannot keep up with "
        "the requested arrival rate. Lower request_rate or use a faster GPU."
    )


def summarize_run(result: SimulationResult) -> RunSummary | None:
    n = len(result.finished)
    if n == 0:
        return None

    finished_sorted = sorted(result.finished, key=lambda r: r.arrival_s)
    warmup_skip = int(n * WARMUP_FRACTION)
    measured = finished_sorted[warmup_skip:] or finished_sorted

    ttfts_ms = [(r.first_token_s - r.arrival_s) * 1000 for r in measured]
    waits_ms = [(r.start_s - r.arrival_s) * 1000 for r in measured]
    e2es_s = [r.finish_s - r.arrival_s for r in measured]
    tpots_ms = [
        ((r.finish_s - r.first_token_s) / max(r.gen_tokens - 1, 1)) * 1000
        for r in measured
    ]

    served_rate = (n / result.wall_time_s) if result.wall_time_s > 0 else 0.0
    saturated, reason = _saturation_reason(result, served_rate)

    n_steps = max(result.n_forward_passes, 1)
    gib = 1024 ** 3
    return RunSummary(
        requested_rate=result.requested_rate,
        served_rate=served_rate,
        n_finished=n,
        ttft_ms=Percentiles.of(ttfts_ms),
        tpot_ms=Percentiles.of(tpots_ms),
        e2e_s=Percentiles.of(e2es_s),
        wait_ms=Percentiles.of(waits_ms),
        wall_time_s=result.wall_time_s,
        mean_in_flight=result.mean_in_flight,
        peak_in_flight=result.peak_in_flight,
        kv_budget_gib=result.kv_budget_bytes / gib,
        peak_kv_use_gib=result.peak_kv_use_bytes / gib,
        admitted_kv_blocked=result.admitted_kv_blocked,
        saturated=saturated,
        saturation_reason=reason,
        n_forward_passes=result.n_forward_passes,
        avg_batch_tokens=result.total_batch_tokens / n_steps,
        avg_op_per_step=_avg_breakdown(result.sim_breakdown, n_steps),
    )


def _bottleneck_label(op: OperationLatency) -> str:
    if op.compute_s > op.memory_s:
        return "COMPUTE"
    if op.memory_s > op.compute_s:
        return "MEMORY"
    return "BALANCED"


def _fmt_pct(p: Percentiles, suffix: str = "ms") -> str:
    return (
        f"mean {p.mean:.2f} {suffix}  "
        f"p50 {p.p50:.2f}  p95 {p.p95:.2f}  p99 {p.p99:.2f}"
    )


def render_report(summary: RunSummary | None) -> str:
    rb = ReportBuilder(width=92)
    rb.line().banner("SERVING SIMULATION REPORT").line()

    if summary is None:
        rb.line("No requests finished to analyze.")
        return rb.build()

    if summary.saturated:
        rb.line(f"⚠  SATURATED — {summary.saturation_reason}")
        rb.line()

    rb.heading("Workload & Throughput")
    rb.kv("Requested rate", f"{summary.requested_rate:.2f} req/s")
    rb.kv("Served rate", f"{summary.served_rate:.2f} req/s")
    rb.kv("Requests completed", f"{summary.n_finished:,}")
    rb.kv("Wall clock", f"{summary.wall_time_s:.2f} s")
    rb.line()

    rb.heading("Latency (warmup excluded)")
    rb.kv("TTFT", _fmt_pct(summary.ttft_ms))
    rb.kv("  wait", _fmt_pct(summary.wait_ms))
    rb.kv("TPOT", _fmt_pct(summary.tpot_ms))
    rb.kv("E2E", _fmt_pct(summary.e2e_s, suffix="s "))
    rb.line()

    rb.heading("System State")
    rb.kv("Mean in-flight", f"{summary.mean_in_flight:.1f} requests")
    rb.kv("Peak in-flight", f"{summary.peak_in_flight} requests")
    rb.kv("KV cache budget", f"{summary.kv_budget_gib:.2f} GiB")
    kv_pressure = (
        summary.peak_kv_use_gib / summary.kv_budget_gib * 100
        if summary.kv_budget_gib > 0 else 0.0
    )
    rb.kv("KV cache peak use", f"{summary.peak_kv_use_gib:.2f} GiB ({kv_pressure:.1f}%)")
    if summary.admitted_kv_blocked > 0:
        rb.kv("KV-blocked admits", f"{summary.admitted_kv_blocked} scheduler steps")

    rb.section("PER-FORWARD-PASS BREAKDOWN")
    step_total = summary.avg_op_per_step.total()
    rb.line(f"Forward passes simulated   : {summary.n_forward_passes:,} steps")
    rb.line(f"Avg tokens / batch         : {summary.avg_batch_tokens:.1f} tokens")
    rb.line(f"Avg latency / forward pass : {step_total.roofline_s * 1000:.3f} ms")
    rb.line()

    pct = lambda v, t: (v / t * 100) if t > 0 else 0.0  # noqa: E731
    rows: list[list[str]] = []
    for name, op in summary.avg_op_per_step.iter_ops():
        rows.append([
            name,
            f"{op.roofline_s * 1000:.3f}",
            f"{pct(op.roofline_s, step_total.roofline_s):.2f}%",
            f"{op.compute_s * 1000:.3f}",
            f"{op.memory_s * 1000:.3f}",
            _bottleneck_label(op),
        ])
    rows.append([
        "TOTAL",
        f"{step_total.roofline_s * 1000:.3f}",
        "100.00%",
        f"{step_total.compute_s * 1000:.3f}",
        f"{step_total.memory_s * 1000:.3f}",
        _bottleneck_label(step_total),
    ])
    rb.table(
        headers=["Op", "Roofline(ms)", "% Step", "Compute(ms)", "Memory(ms)", "Bottleneck"],
        rows=rows,
        col_widths=[14, 12, 8, 11, 10, 10],
    )

    rb.rule("=")
    return rb.build()


# ---------------------------------------------------------------------------
# Tool: simulate_serving
# ---------------------------------------------------------------------------

@tool(
    "Simulate a continuous-batching serving workload under a Poisson "
    "arrival process and report TTFT/TPOT/E2E percentiles, observed "
    "concurrency, KV-cache pressure, per-forward-pass breakdown, and a "
    "saturation flag if the GPU(s) cannot keep up with the requested rate. "
    "Concurrency is a RESULT, not an input — it emerges from request_rate, "
    "request sizes, and serving capacity. The workload knobs all come from "
    "a WorkloadProfile YAML (typically `stages/01_workload.yaml`); pass "
    "the hardware as `gpu`.",
    workload_file="Workspace-relative path to a WorkloadProfile YAML "
                  "(e.g. 'stages/01_workload.yaml'). Must contain at "
                  "least `model`, `request_rate`, `input_len`, "
                  "`output_len`, `num_requests`, `max_num_batched_tokens`. "
                  "`max_concurrent_requests` is optional (defaults to 1024).",
    gpu="Preset GPU name (must exist in PRESET_GPUS).",
    n_gpus="Number of GPUs sharing the model (for KV-cache budget, default 1).",
)
def simulate_serving(
    workload_file: str,
    gpu: str,
    n_gpus: int = 1,
) -> str:
    try:
        from ...workspace import resolve
        import yaml
        wf = yaml.safe_load(resolve(workload_file).read_text()) or {}
    except Exception as e:
        return f"ERROR: could not load workload_file {workload_file!r}: {type(e).__name__}: {e}"

    model = wf.get("model", "")
    request_rate = wf.get("request_rate", 0.0)
    input_len = wf.get("input_len", 0)
    output_len = wf.get("output_len", 0)
    num_requests = wf.get("num_requests", 0)
    max_num_batched_tokens = wf.get("max_num_batched_tokens", 0)
    max_concurrent_requests = wf.get("max_concurrent_requests", 0) or 1024
    range_ratio = float(wf.get("range_ratio", 0.0))

    if model not in PRESET_MODELS:
        return f"ERROR: unknown model {model!r}. Available: {', '.join(sorted(PRESET_MODELS))}"
    if gpu not in PRESET_GPUS:
        return f"ERROR: unknown gpu {gpu!r}. Available: {', '.join(sorted(PRESET_GPUS))}"
    if request_rate <= 0:
        return "ERROR: request_rate must be > 0."

    result = run_simulation(
        model_name=model,
        gpu_name=gpu,
        request_rate=request_rate,
        input_len=input_len,
        output_len=output_len,
        n_requests=num_requests,
        max_batched_tokens=max_num_batched_tokens,
        max_concurrent_requests=max_concurrent_requests,
        n_gpus=n_gpus,
        jitter=range_ratio,
    )
    return render_report(summarize_run(result))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate a continuous-batching serving workload under "
                    "Poisson arrivals. Workload knobs (model, request_rate, "
                    "input/output_len, num_requests, batched-tokens, "
                    "max-concurrent) come from a WorkloadProfile YAML."
    )
    parser.add_argument("--workload-file", type=str, required=True,
                        help="Path to a WorkloadProfile YAML (relative to "
                             "CWD or absolute). Must contain model, "
                             "request_rate, input_len, output_len, "
                             "num_requests, max_num_batched_tokens. "
                             "max_concurrent_requests is optional.")
    parser.add_argument("--gpu", type=str, required=True,
                        help="Preset GPU name (PRESET_GPUS key).")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench-file", type=str, default=None,
                        help="Microbenchmark results JSON (optional).")
    return parser.parse_args()


if __name__ == "__main__":
    import yaml as _yaml
    from pathlib import Path as _Path
    args = _parse_args()
    microbench = None
    if args.bench_file:
        with open(args.bench_file) as f:
            microbench = json.load(f)

    wf = _yaml.safe_load(_Path(args.workload_file).read_text()) or {}
    print(f"running workload simulation from {args.workload_file}...")
    result = run_simulation(
        model_name=wf["model"],
        gpu_name=args.gpu,
        request_rate=wf["request_rate"],
        input_len=wf["input_len"],
        output_len=wf["output_len"],
        n_requests=wf["num_requests"],
        max_batched_tokens=wf["max_num_batched_tokens"],
        max_concurrent_requests=wf.get("max_concurrent_requests", 1024),
        n_gpus=args.n_gpus,
        jitter=float(wf.get("range_ratio", 0.0)),
        microbench=microbench,
        seed=args.seed,
    )
    print(render_report(summarize_run(result)))
