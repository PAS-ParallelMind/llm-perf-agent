"""Continuous-batching serving simulator with Poisson-arrival workload.

The workload is described by an arrival *rate* (req/s, Poisson) plus
per-request (input_len, output_len). Concurrency is then a *result* — it
emerges from the arrival rate, request size, and serving capacity:

* ``max_concurrent_requests`` is a server-policy cap (vLLM ``--max-num-seqs``).
* The KV cache budget caps concurrency further: a request is admitted
  only if its worst-case future KV (prompt+gen tokens, summed across
  layers with SWA caps) fits alongside currently-running requests.

Exposed as the ``simulate_serving`` tool; also runnable as a CLI for
deeper exploration (microbenchmark file support, custom seed, etc.).
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass

from ..base import tool
from .configs.hw_specs import PRESET_GPUS
from .configs.model_specs import PRESET_MODELS, ModelConfig, get_quantization_bytes
from .latency import (
    MODES,
    ParallelConfig,
    Request,
    StepBreakdown,
    forward_pass_latency,
    total_tokens_in_batch,
)
from .memory import weights_vram_gib
from .report import ReportBuilder


# Fraction of GPU memory reserved for activations / CUDA graphs / framework
# overhead. Matches vLLM's default ``--gpu-memory-utilization 0.92``, i.e.
# 8% of total HBM is held back from the KV cache budget.
GPU_MEMORY_OVERHEAD = 0.08

# Discard the first WARMUP_FRACTION of completed requests when computing
# tail-latency percentiles — they finish before the system reaches steady
# state, so their TTFTs are artificially low.
WARMUP_FRACTION = 0.10


# ---------------------------------------------------------------------------
# KV budget accounting
# ---------------------------------------------------------------------------

def _kv_bytes_per_token_summed(
    model: ModelConfig, context_tokens: int, tp: int = 1,
) -> int:
    """Worst-case cluster-wide KV bytes for one request that grows to
    ``context_tokens``.

    Sums per-layer cost respecting sliding-window caps. When ``tp >
    n_kv_heads`` each KV head is replicated across ``tp / n_kv_heads``
    ranks so the cluster-wide KV cache scales by that ratio — modeled by
    using ``max(n_kv_heads, tp)`` as the effective head count.
    """
    kv_byte = get_quantization_bytes(model.kv_cache_dtype)
    effective_n_kv = max(model.n_kv_heads, tp)
    bytes_per_token_per_layer = 2 * effective_n_kv * model.head_dim * kv_byte
    tokens_summed = sum(
        model.effective_kv_tokens(layer_idx, context_tokens)
        for layer_idx in range(model.n_layers)
    )
    return bytes_per_token_per_layer * tokens_summed


def compute_kv_budget_bytes(model: ModelConfig, gpu, parallel: ParallelConfig) -> int:
    """Per-replica KV-cache budget in bytes after weights + overhead.

    Each TP rank holds 1/TP of the model weights, so the replica's
    aggregate budget is ``tp * per_gpu_mem - full_weights - tp * overhead``.
    DP is not factored in: replicas are independent, so the budget is
    reported per replica.
    """
    total_bytes = int(gpu.mem_capacity * parallel.tp)
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
    tp: int = 1

    def _request_worst_case_kv_bytes(self, r: Request) -> int:
        return _kv_bytes_per_token_summed(
            self.model, r.prompt_tokens + r.gen_tokens, self.tp,
        )

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
    """Everything the report needs from one simulation run.

    The simulator only models ONE replica; aggregate (system-level)
    rates are computed at report time by scaling by ``dp``. The
    ``requested_rate`` here is the system-level rate the user asked for,
    so saturation comparisons stay apples-to-apples.
    """
    finished: list[Request]
    sim_breakdown: StepBreakdown
    n_forward_passes: int
    total_batch_tokens: int
    wall_time_s: float
    requested_rate: float          # system-level (sum across DP replicas)
    n_requests: int                # system-level
    parallel: ParallelConfig

    # System instrumentation (per-replica)
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


def _request_lengths(
    rng: random.Random,
    n_requests: int,
    input_len: int,
    output_len: int,
    jitter: float,
) -> list[tuple[int, int]]:
    """Per-request (prompt_tokens, gen_tokens) pairs with jitter applied.

    Factored out so open-loop and closed-loop modes can share the same
    jitter semantics — open-loop pairs these with Poisson arrivals up-front,
    closed-loop dispatches them on-demand as slots free.
    """
    out: list[tuple[int, int]] = []
    for _ in range(n_requests):
        prompt = max(1, int(rng.uniform(input_len * (1 - jitter), input_len * (1 + jitter))))
        gen = max(1, int(rng.uniform(output_len * (1 - jitter), output_len * (1 + jitter))))
        out.append((prompt, gen))
    return out


def _build_request_pool(
    rng: random.Random,
    n_requests: int,
    input_len: int,
    output_len: int,
    request_rate: float,
    jitter: float,
) -> list[Request]:
    """Open-loop pool: Poisson arrivals + per-request lengths."""
    arrivals = _poisson_arrivals(rng, n_requests, request_rate)
    lengths = _request_lengths(rng, n_requests, input_len, output_len, jitter)
    return [
        Request(id=i, arrival_s=a, prompt_tokens=p, gen_tokens=g)
        for i, (a, (p, g)) in enumerate(zip(arrivals, lengths))
    ]


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
    parallel: ParallelConfig = ParallelConfig(),
    jitter: float = 0.1,
    seed: int = 0,
    progress_cb=None,
    latency_source: str = "baseline",
) -> SimulationResult:
    """Run the simulation. Wall clock advances by each step's modeled time.

    Two arrival modes (auto-detected from ``request_rate``):

    * **Open-loop** (finite ``request_rate``): arrivals follow a Poisson
      process pre-generated up front. Concurrency emerges from rate and
      service time.
    * **Closed-loop** (``request_rate=math.inf``): seeds
      ``max_concurrent_requests`` requests at t=0, then dispatches the next
      request *as each one finishes* — steady-state in-flight = N. Use for
      kernel-efficiency calibration where you want a controlled batch size.

    Parallelism: ``tp`` shards each layer's matmuls/attention heads
    across TP ranks and adds two ring all-reduces per layer (see
    ``latency._transformer_layer_s``). ``dp`` replicates the model
    across independent serving groups — the simulator runs ONE replica
    with ``request_rate/dp`` arrivals (open-loop) or ``n_requests/dp``
    requests (closed-loop) and the report scales rates/throughput by
    ``dp`` at the system level. ``max_concurrent_requests`` is the
    **per-replica** cap. Total GPU count is ``tp * dp``.

    Two latency sources are supported via ``latency_source``:

    * ``"baseline"`` (default) — per-shape latency from the llm-gpu-bench
      microbench grid (under ``configs/hw_profiles/<gpu>/``). This is
      the realistic projection — reproduces measured TPOT within ~15%
      on B200 across N=1..256.
    * ``"theoretical"`` — analytic FLOPs/bytes against the GPU's
      theoretical peaks (efficiency = 1.0 for every op). The optimistic
      ceiling; over-predicts throughput by ~5-10× on real hardware.
    """
    if latency_source not in MODES:
        raise ValueError(
            f"latency_source={latency_source!r} must be one of {MODES}")
    rng = random.Random(seed)
    model = PRESET_MODELS[model_name]
    gpu = PRESET_GPUS[gpu_name]
    parallel.validate(model)
    kv_budget = compute_kv_budget_bytes(model, gpu, parallel)
    scheduler = ContinuousBatchingScheduler(
        max_batched_tokens=max_batched_tokens,
        max_concurrent_requests=max_concurrent_requests,
        kv_budget_bytes=kv_budget,
        model=model,
        tp=parallel.tp,
    )

    closed_loop = math.isinf(request_rate)
    # DP shards the workload across replicas; we simulate one of them.
    # System-level rate / count stay in the result for reporting.
    sys_request_rate = request_rate
    sys_n_requests = n_requests
    per_replica_n_requests = max(1, n_requests // parallel.dp)
    per_replica_rate = request_rate if closed_loop else (request_rate / parallel.dp)

    if closed_loop:
        # Closed-loop: pre-generate lengths only; arrivals fire on-demand as
        # requests finish, keeping in-flight at most max_concurrent_requests.
        lengths = _request_lengths(rng, per_replica_n_requests, input_len, output_len, jitter)
        arrival_pool: list[Request] = []
        initial = min(max_concurrent_requests, per_replica_n_requests)
        for i in range(initial):
            p, g = lengths[i]
            arrival_pool.append(Request(id=i, arrival_s=0.0, prompt_tokens=p, gen_tokens=g))
        next_dispatch_idx = initial
    else:
        arrival_pool = _build_request_pool(
            rng, per_replica_n_requests, input_len, output_len, per_replica_rate, jitter,
        )
        lengths = None
        next_dispatch_idx = per_replica_n_requests  # all already in arrival_pool

    waiting_queue: list[Request] = []
    running_queue: list[Request] = []
    finished: list[Request] = []

    sim_breakdown = StepBreakdown()
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
            model, r.prompt_tokens + r.gen_tokens, parallel.tp,
        ))

    pool_idx = 0
    while (pool_idx < len(arrival_pool) or waiting_queue or running_queue
           or (closed_loop and next_dispatch_idx < per_replica_n_requests)):
        # Pull arrivals up to t_now into the waiting queue.
        while pool_idx < len(arrival_pool) and arrival_pool[pool_idx].arrival_s <= t_now:
            waiting_queue.append(arrival_pool[pool_idx])
            pool_idx += 1

        # If running is empty and waiting is empty, jump the clock to the
        # next arrival (no work to do until then). Only relevant in open-loop;
        # closed-loop always has work pending or has finished entirely.
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

        # Forward pass — single path, latency_source picks the mode.
        step_breakdown = forward_pass_latency(
            running_queue, gpu_name, model,
            mode=latency_source, parallel=parallel,
        )
        sim_breakdown.accumulate(step_breakdown)
        step_duration = step_breakdown.total_s
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
                # Closed-loop: a slot just freed — dispatch the next request.
                if closed_loop and next_dispatch_idx < per_replica_n_requests:
                    p, g = lengths[next_dispatch_idx]
                    arrival_pool.append(Request(
                        id=next_dispatch_idx,
                        arrival_s=t_now,
                        prompt_tokens=p,
                        gen_tokens=g,
                    ))
                    next_dispatch_idx += 1
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
        requested_rate=sys_request_rate,
        n_requests=sys_n_requests,
        parallel=parallel,
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
    # Workload (system-level: aggregated across DP replicas)
    requested_rate: float
    served_rate: float
    output_token_throughput: float   # generated tokens per second (aggregate)
    n_finished: int

    # Parallelism strategy used for the run
    parallel: ParallelConfig

    # Latency percentiles (ms / s) — identical across replicas, so reported as-is
    ttft_ms: Percentiles
    tpot_ms: Percentiles
    e2e_s: Percentiles
    wait_ms: Percentiles

    # System (aggregated across DP replicas — whole-system view)
    wall_time_s: float
    mean_in_flight: float
    peak_in_flight: int
    kv_budget_gib: float
    peak_kv_use_gib: float
    admitted_kv_blocked: int

    # Per-forward-pass averages
    n_forward_passes: int
    avg_batch_tokens: float
    avg_op_per_step: StepBreakdown


def _avg_breakdown(breakdown: StepBreakdown, n: int) -> StepBreakdown:
    return StepBreakdown(
        qkv_s=breakdown.qkv_s / n,
        attn_decode_s=breakdown.attn_decode_s / n,
        attn_prefill_s=breakdown.attn_prefill_s / n,
        o_proj_s=breakdown.o_proj_s / n,
        ffn_or_moe_s=breakdown.ffn_or_moe_s / n,
        comm_s=breakdown.comm_s / n,
        lm_head_s=breakdown.lm_head_s / n,
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

    # DP replicas serve independently in parallel, so system-level rates
    # scale linearly with dp. Per-request latencies are identical on each
    # replica and don't need scaling. n_finished is reported as the
    # aggregate count across replicas (per-replica count × dp).
    dp = result.parallel.dp
    per_replica_rate = (n / result.wall_time_s) if result.wall_time_s > 0 else 0.0
    served_rate = per_replica_rate * dp
    total_gen_tokens = sum(r.gen_tokens for r in result.finished)
    per_replica_tps = (
        total_gen_tokens / result.wall_time_s if result.wall_time_s > 0 else 0.0
    )
    output_token_throughput = per_replica_tps * dp

    # System-state aggregates: every replica runs the same workload, so
    # the system-level value is just the per-replica value × dp. Per-op
    # forward-pass timings stay per-replica (they're "what one GPU group
    # does in one step", not a system-wide aggregate).
    n_steps = max(result.n_forward_passes, 1)
    gib = 1024 ** 3
    return RunSummary(
        requested_rate=result.requested_rate,
        served_rate=served_rate,
        output_token_throughput=output_token_throughput,
        n_finished=n * dp,
        parallel=result.parallel,
        ttft_ms=Percentiles.of(ttfts_ms),
        tpot_ms=Percentiles.of(tpots_ms),
        e2e_s=Percentiles.of(e2es_s),
        wait_ms=Percentiles.of(waits_ms),
        wall_time_s=result.wall_time_s,
        mean_in_flight=result.mean_in_flight * dp,
        peak_in_flight=result.peak_in_flight * dp,
        kv_budget_gib=result.kv_budget_bytes / gib * dp,
        peak_kv_use_gib=result.peak_kv_use_bytes / gib * dp,
        admitted_kv_blocked=result.admitted_kv_blocked * dp,
        n_forward_passes=result.n_forward_passes,
        avg_batch_tokens=result.total_batch_tokens / n_steps,
        avg_op_per_step=_avg_breakdown(result.sim_breakdown, n_steps),
    )


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

    KEY_WIDTH = 32

    rb.heading("Workload & Throughput")
    p = summary.parallel
    rb.kv("Parallelism",             f"tp={p.tp} dp={p.dp}  ({p.n_gpus} GPUs total)", KEY_WIDTH)
    rb.kv("Incoming request rate",   f"{summary.requested_rate:.2f} req/s", KEY_WIDTH)
    rb.kv("Completed request rate",  f"{summary.served_rate:.2f} req/s", KEY_WIDTH)
    rb.kv("Output token throughput", f"{summary.output_token_throughput:.1f} tok/s", KEY_WIDTH)
    rb.kv("Requests completed",      f"{summary.n_finished:,}", KEY_WIDTH)
    rb.kv("Simulated duration",      f"{summary.wall_time_s:.2f} s", KEY_WIDTH)
    rb.line()

    rb.heading("Per-Request Latency (warmup excluded)")
    rb.kv("Time to First Token (TTFT)",   _fmt_pct(summary.ttft_ms), KEY_WIDTH)
    rb.kv("  Queue wait time",            _fmt_pct(summary.wait_ms), KEY_WIDTH)
    rb.kv("Time Per Output Token (TPOT)", _fmt_pct(summary.tpot_ms), KEY_WIDTH)
    rb.kv("End-to-end latency (E2E)",     _fmt_pct(summary.e2e_s, suffix="s "), KEY_WIDTH)
    rb.line()

    rb.heading("System State")
    rb.kv("Avg concurrent requests",  f"{summary.mean_in_flight:.1f} requests", KEY_WIDTH)
    rb.kv("Peak concurrent requests", f"{summary.peak_in_flight} requests", KEY_WIDTH)
    rb.kv("KV cache budget",          f"{summary.kv_budget_gib:.2f} GiB", KEY_WIDTH)
    kv_pressure = (
        summary.peak_kv_use_gib / summary.kv_budget_gib * 100
        if summary.kv_budget_gib > 0 else 0.0
    )
    rb.kv("KV cache peak use",        f"{summary.peak_kv_use_gib:.2f} GiB ({kv_pressure:.1f}%)", KEY_WIDTH)
    if summary.admitted_kv_blocked > 0:
        rb.kv("Admission stalls (KV full)", f"{summary.admitted_kv_blocked} scheduler steps", KEY_WIDTH)

    rb.section("PER-FORWARD-PASS BREAKDOWN")
    step_total_s = summary.avg_op_per_step.total_s
    rb.line(f"Forward passes simulated     : {summary.n_forward_passes:,} steps")
    rb.line(f"Avg batch size (tokens/pass) : {summary.avg_batch_tokens:.1f} tokens")
    rb.line(f"Avg latency per forward pass : {step_total_s * 1000:.3f} ms")
    rb.line()

    pct = lambda v, t: (v / t * 100) if t > 0 else 0.0  # noqa: E731
    rows: list[list[str]] = []
    for name, op_s in summary.avg_op_per_step.iter_ops():
        rows.append([
            name,
            f"{op_s * 1000:.3f}",
            f"{pct(op_s, step_total_s):.2f}%",
        ])
    rows.append([
        "TOTAL",
        f"{step_total_s * 1000:.3f}",
        "100.00%",
    ])
    rb.table(
        headers=["Op", "Time(ms)", "% Step"],
        rows=rows,
        col_widths=[14, 12, 10],
    )

    rb.rule("=")
    return rb.build()


# ---------------------------------------------------------------------------
# Tool: simulate_serving
# ---------------------------------------------------------------------------

@tool(
    "Simulate a continuous-batching serving workload and report TTFT/TPOT/"
    "E2E percentiles, observed concurrency, KV-cache pressure, and per-"
    "forward-pass breakdown. Auto-detects two modes from the workload's "
    "`request_rate`: a finite rate runs OPEN-loop (Poisson arrivals, "
    "concurrency is a result); `.inf` runs CLOSED-loop (a new request is "
    "dispatched as each one finishes, so steady-state in-flight = "
    "`max_concurrent_requests`). Open-loop is for deployment-capacity "
    "sizing; closed-loop is for kernel-efficiency calibration at a "
    "controlled batch. The workload knobs all come from a WorkloadProfile "
    "YAML; pass the hardware as `gpu`.",
    workload_file="Workspace-relative path to a WorkloadProfile YAML "
                  "(e.g. 'stages/01_workload.yaml'). Must contain at "
                  "least `model`, `request_rate` (finite or `.inf`), "
                  "`input_len`, `output_len`, `num_requests`, "
                  "`max_num_batched_tokens`. `max_concurrent_requests` is "
                  "optional in open-loop (defaults to 1024) and REQUIRED "
                  "in closed-loop (it sets the steady-state in-flight N).",
    gpu="Preset GPU name (must exist in PRESET_GPUS).",
    tp="Tensor-parallel degree per replica (must divide model head counts; "
       "default 1). Adds two ring all-reduces per layer modeled against "
       "the GPU's NVLink bandwidth.",
    dp="Data-parallel replication factor (default 1). Independent replicas; "
       "throughput and served rate scale linearly. Total GPUs = tp * dp.",
    latency_source="How to model per-forward-pass time. 'baseline' (default) "
                   "uses the per-shape microbench grid for this GPU "
                   "(configs/hw_profiles/<gpu>/*.json) — the realistic "
                   "projection. 'theoretical' uses analytic FLOPs/bytes "
                   "× the GPU's theoretical peaks (efficiency=1.0 for "
                   "every op) — the optimistic ceiling.",
)
def simulate_serving(
    workload_file: str,
    gpu: str,
    tp: int = 1,
    dp: int = 1,
    latency_source: str = "baseline",
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
    try:
        rate_f = float(request_rate)
    except (TypeError, ValueError):
        return f"ERROR: request_rate={request_rate!r} is not numeric (use a positive float, or `.inf` for closed-loop)."
    if not (rate_f > 0):
        return "ERROR: request_rate must be > 0 (or `.inf` for closed-loop)."
    if latency_source not in MODES:
        return f"ERROR: latency_source={latency_source!r} must be one of {MODES}."

    try:
        parallel = ParallelConfig(tp=tp, dp=dp)
        parallel.validate(PRESET_MODELS[model])
    except ValueError as e:
        return f"ERROR: invalid parallelism: {e}"

    result = run_simulation(
        model_name=model,
        gpu_name=gpu,
        request_rate=rate_f,
        input_len=input_len,
        output_len=output_len,
        n_requests=num_requests,
        max_batched_tokens=max_num_batched_tokens,
        max_concurrent_requests=max_concurrent_requests,
        parallel=parallel,
        jitter=range_ratio,
        latency_source=latency_source,
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
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor-parallel degree per replica (default 1).")
    parser.add_argument("--dp", type=int, default=1,
                        help="Data-parallel replication factor (default 1).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency-source", type=str, default="baseline",
                        choices=list(MODES),
                        help="Per-pass latency source. 'baseline' (default) "
                             "uses the per-shape microbench grid (realistic "
                             "projection). 'theoretical' uses analytic peaks "
                             "with all efficiency factors = 1 (optimistic "
                             "ceiling).")
    return parser.parse_args()


if __name__ == "__main__":
    import yaml as _yaml
    from pathlib import Path as _Path
    args = _parse_args()

    wf = _yaml.safe_load(_Path(args.workload_file).read_text()) or {}
    print(f"running workload simulation from {args.workload_file}...")
    result = run_simulation(
        model_name=wf["model"],
        gpu_name=args.gpu,
        request_rate=float(wf["request_rate"]),
        input_len=wf["input_len"],
        output_len=wf["output_len"],
        n_requests=wf["num_requests"],
        max_batched_tokens=wf["max_num_batched_tokens"],
        max_concurrent_requests=wf.get("max_concurrent_requests", 1024),
        parallel=ParallelConfig(tp=args.tp, dp=args.dp),
        jitter=float(wf.get("range_ratio", 0.0)),
        seed=args.seed,
        latency_source=args.latency_source,
    )
    print(render_report(summarize_run(result)))
