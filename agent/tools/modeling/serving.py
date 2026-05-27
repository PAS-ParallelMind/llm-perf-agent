"""Continuous-batching serving simulator.

Schedules a fixed request pool through a vLLM-style scheduler under a
max-batched-tokens budget, accumulates per-request and per-phase
latencies, and reports TTFT / TPOT / throughput plus a
compute-vs-memory bottleneck breakdown.

Exposed as the ``simulate_serving`` tool; also runnable as a CLI for
deeper exploration (with microbenchmark file support, custom seed, etc.).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from ..base import tool
from .configs.hw_specs import PRESET_GPUS
from .configs.model_specs import PRESET_MODELS
from .latency import (
    OperationLatency,
    OpBreakdown,
    Request,
    forward_pass_latency,
    total_tokens_in_batch,
)
from .report import ReportBuilder


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class ContinuousBatchingScheduler:
    """vLLM-style continuous-batching scheduler.

    Each :meth:`schedule_step` call assigns ``tokens_this_step`` to every
    running request and may admit fresh requests up to ``concurrency``.

    Priority order, matching vLLM's defaults:
      1. Decode tokens for currently running requests (1 token each).
      2. Prefill chunks for running requests still in their prompt.
      3. Admit waiting-queue requests from the request pool.
      4. Start prefill on waiting requests until budget/concurrency exhausted.
    """

    max_batched_tokens: int
    concurrency: int

    def schedule_step(
        self,
        request_pool: list[Request],
        waiting_queue: list[Request],
        running_queue: list[Request],
    ) -> None:
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

        # 4) admit fresh requests from the pool into the waiting queue.
        while (
            len(running_queue) + len(waiting_queue) < self.concurrency
            and request_pool
        ):
            waiting_queue.append(request_pool.pop(0))

        # 5) start prefill on waiting requests until budget/concurrency exhausted.
        while (
            total_tokens_in_batch(running_queue) < self.max_batched_tokens
            and waiting_queue
            and len(running_queue) < self.concurrency
        ):
            r = waiting_queue.pop(0)
            budget_left = self.max_batched_tokens - total_tokens_in_batch(running_queue)
            r.tokens_this_step = min(r.prompt_tokens, budget_left)
            running_queue.append(r)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def _build_request_pool(
    rng: random.Random,
    n_requests: int,
    input_len: int,
    output_len: int,
    jitter: float,
) -> list[Request]:
    """Build n_requests with input/output length jittered by ±jitter ratio."""
    pool: list[Request] = []
    for i in range(n_requests):
        prompt = int(rng.uniform(input_len * (1 - jitter), input_len * (1 + jitter)))
        gen = int(rng.uniform(output_len * (1 - jitter), output_len * (1 + jitter)))
        pool.append(Request(id=i, arrival_s=0.0, prompt_tokens=prompt, gen_tokens=gen))
    return pool


def _request_is_done(r: Request) -> bool:
    """A request finishes when its KV cache holds prompt+gen-1 tokens.

    The last generated token is produced but not cached (no further decode
    will read it), matching the original vLLM-style accounting.
    """
    return r.kv_tokens == r.prompt_tokens + r.gen_tokens - 1


@dataclass
class SimulationResult:
    """Everything the report needs from one simulation run."""
    finished: list[Request]
    sim_breakdown: OpBreakdown   # totals across all forward passes
    n_forward_passes: int        # number of scheduler steps executed
    total_batch_tokens: int      # sum of batch sizes across all steps


def run_simulation(
    model_name: str,
    gpu_name: str,
    concurrency: int,
    input_len: int,
    output_len: int,
    n_requests: int,
    max_batched_tokens: int,
    jitter: float = 0.1,
    microbench: dict | None = None,
    seed: int = 0,
    progress_cb=None,
) -> SimulationResult:
    """Run the scheduler simulation and return aggregated counters."""
    rng = random.Random(seed)
    model = PRESET_MODELS[model_name]
    gpu = PRESET_GPUS[gpu_name]
    scheduler = ContinuousBatchingScheduler(
        max_batched_tokens=max_batched_tokens,
        concurrency=concurrency,
    )

    request_pool = _build_request_pool(rng, n_requests, input_len, output_len, jitter)
    waiting_queue: list[Request] = []
    running_queue: list[Request] = []
    finished_queue: list[Request] = []

    sim_breakdown = OpBreakdown()
    n_forward_passes = 0
    total_batch_tokens = 0
    # Cache per-step MoE expert-load samples so we don't redo the random
    # sampling for repeated batch sizes. Scoped to this simulation only.
    expert_cache: dict = {}

    while request_pool or running_queue:
        scheduler.schedule_step(request_pool, waiting_queue, running_queue)
        step_breakdown = forward_pass_latency(running_queue, microbench, gpu, model, expert_cache)
        sim_breakdown.accumulate(step_breakdown)
        n_forward_passes += 1
        total_batch_tokens += total_tokens_in_batch(running_queue)

        still_running: list[Request] = []
        for r in running_queue:
            r.kv_tokens += r.tokens_this_step
            if r.prompt_tokens >= r.kv_tokens:
                r.latency.prefill.accumulate(step_breakdown)
            else:
                r.latency.decode.accumulate(step_breakdown)
            if _request_is_done(r):
                finished_queue.append(r)
                if progress_cb is not None:
                    progress_cb(1)
            else:
                still_running.append(r)
        running_queue[:] = still_running

        step_total = step_breakdown.total()
        for r in waiting_queue:
            r.latency.waiting += step_total

    return SimulationResult(
        finished=finished_queue,
        sim_breakdown=sim_breakdown,
        n_forward_passes=n_forward_passes,
        total_batch_tokens=total_batch_tokens,
    )


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    # Per-request stats (aggregated across finished requests)
    n_finished: int
    avg_waiting_s: float
    avg_prefill_s: float
    avg_decode_per_token_s: float
    avg_request_s: float

    output_throughput_tps: float
    total_throughput_tps: float

    # System-level stats (averaged across forward-pass steps)
    n_forward_passes: int
    avg_batch_tokens: float           # mean tokens-in-batch per forward pass
    avg_op_per_step: OpBreakdown      # per-op latency averaged over all steps


def _avg_op(op: OperationLatency, n: int) -> OperationLatency:
    return OperationLatency(op.compute_s / n, op.memory_s / n, op.roofline_s / n)


def _avg_breakdown(breakdown: OpBreakdown, n: int) -> OpBreakdown:
    avg = OpBreakdown()
    for name, op in breakdown.iter_ops():
        setattr(avg, name, _avg_op(op, n))
    return avg


def summarize_run(result: SimulationResult) -> RunSummary | None:
    """Aggregate per-request and per-step counters into the report summary."""
    n = len(result.finished)
    if n == 0:
        return None

    sum_waiting_s = 0.0
    sum_prefill_total = 0.0
    sum_decode_per_token = 0.0
    sum_total_latency = 0.0
    sum_prompt_plus_gen_tokens = 0
    sum_gen_tokens = 0

    for r in result.finished:
        prefill_total = r.latency.prefill.total().roofline_s
        decode_total = r.latency.decode.total().roofline_s
        sum_waiting_s += r.latency.waiting.roofline_s
        sum_prefill_total += prefill_total
        if r.gen_tokens > 1:
            sum_decode_per_token += decode_total / (r.gen_tokens - 1)

        sum_total_latency += r.latency.waiting.roofline_s + prefill_total + decode_total
        sum_prompt_plus_gen_tokens += r.prompt_tokens + r.gen_tokens
        sum_gen_tokens += r.gen_tokens

    sim_total_s = result.sim_breakdown.total().roofline_s
    n_steps = max(result.n_forward_passes, 1)
    return RunSummary(
        n_finished=n,
        avg_waiting_s=sum_waiting_s / n,
        avg_prefill_s=sum_prefill_total / n,
        avg_decode_per_token_s=sum_decode_per_token / n,
        avg_request_s=sum_total_latency / n,
        output_throughput_tps=(sum_gen_tokens / sim_total_s) if sim_total_s > 0 else 0.0,
        total_throughput_tps=(sum_prompt_plus_gen_tokens / sim_total_s) if sim_total_s > 0 else 0.0,
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


def render_report(summary: RunSummary | None,
                  cost_per_hour: float = 0.0) -> str:
    rb = ReportBuilder(width=76)
    rb.line().banner("SERVING SIMULATION REPORT").line()

    if summary is None:
        rb.line("No requests finished to analyze.")
        return rb.build()

    rb.line(f"Total Requests Processed: {summary.n_finished}")
    rb.line()
    rb.line(f"Average Request Latency:  {summary.avg_request_s:.3f} s")
    rb.line(f"Average TTFT:             {(summary.avg_waiting_s + summary.avg_prefill_s) * 1000:.3f} ms")
    rb.line(f"  - Wait Time:            {summary.avg_waiting_s * 1000:.3f} ms")
    rb.line(f"  - Prefill Time:         {summary.avg_prefill_s * 1000:.3f} ms")
    rb.line(f"Average TPOT:             {summary.avg_decode_per_token_s * 1000:.3f} ms")
    rb.line(f"Output Token Throughput:  {summary.output_throughput_tps:.3f} token/s")
    rb.line(f"Total Token Throughput:   {summary.total_throughput_tps:.3f} token/s")

    # Cost / 1M tokens — only when the GPU has a non-zero price.
    if cost_per_hour > 0.0 and summary.output_throughput_tps > 0:
        cost_per_s = cost_per_hour / 3600.0
        cost_per_mtok_out = cost_per_s * 1e6 / summary.output_throughput_tps
        cost_per_mtok_total = (cost_per_s * 1e6 / summary.total_throughput_tps
                               if summary.total_throughput_tps > 0 else None)
        rb.line(f"Cost / 1M output tokens:  ${cost_per_mtok_out:.3f}  "
                f"(at ${cost_per_hour:.2f}/GPU-hour)")
        if cost_per_mtok_total is not None:
            rb.line(f"Cost / 1M total tokens:   ${cost_per_mtok_total:.3f}")

    pct = lambda v, t: (v / t * 100) if t > 0 else 0.0  # noqa: E731

    rb.section("PER-FORWARD-PASS BREAKDOWN")
    # Continuous batching mixes prefill+decode tokens in every forward pass,
    # so "pure prefill" / "pure decode" averages would be misleading. Show
    # per-op latency averaged over every scheduler step instead.
    step_total = summary.avg_op_per_step.total()
    rb.line(f"Forward passes simulated   : {summary.n_forward_passes:,} steps")
    rb.line(f"Avg tokens / batch         : {summary.avg_batch_tokens:.1f} tokens")
    rb.line(f"Avg latency / forward pass : {step_total.roofline_s * 1000:.3f} ms")
    rb.line()

    step_rows: list[list[str]] = []
    for name, op in summary.avg_op_per_step.iter_ops():
        step_rows.append([
            name,
            f"{op.roofline_s * 1000:.3f}",
            f"{pct(op.roofline_s, step_total.roofline_s):.2f}%",
            f"{op.compute_s * 1000:.3f}",
            f"{op.memory_s * 1000:.3f}",
            _bottleneck_label(op),
        ])
    step_rows.append([
        "TOTAL",
        f"{step_total.roofline_s * 1000:.3f}",
        "100.00%",
        f"{step_total.compute_s * 1000:.3f}",
        f"{step_total.memory_s * 1000:.3f}",
        _bottleneck_label(step_total),
    ])
    rb.table(
        headers=["Op", "Roofline(ms)", "% Step", "Compute(ms)", "Memory(ms)", "Bottleneck"],
        rows=step_rows,
        col_widths=[14, 12, 8, 11, 10, 10],
    )

    # Top bottleneck: the op consuming the largest share of step latency.
    # Continuous batching mixes prefill+decode tokens within each forward
    # pass, so the bottleneck is named per-OP, not per-phase.
    top_name, top_op = max(summary.avg_op_per_step.iter_ops(),
                           key=lambda no: no[1].roofline_s)
    rb.line()
    rb.line(f"Bottleneck: {top_name} — "
            f"{pct(top_op.roofline_s, step_total.roofline_s):.1f}% of step "
            f"({_bottleneck_label(top_op)}-bound, "
            f"avg {summary.avg_batch_tokens:.1f} tokens/batch)")

    rb.rule("=")
    return rb.build()


# ---------------------------------------------------------------------------
# Tool: simulate_serving
# ---------------------------------------------------------------------------

@tool(
    "Simulate a continuous-batching serving workload and report TTFT, "
    "TPOT, throughput, and compute-vs-memory bottlenecks. Models a "
    "fixed request pool flowing through a scheduler with a max-batched-"
    "tokens budget. Use this to compare deployment configurations or "
    "diagnose what's limiting performance.",
    model="Preset model name (must exist in PRESET_MODELS).",
    gpu="Preset GPU name (must exist in PRESET_GPUS).",
    concurrency="Maximum number of concurrent in-flight requests.",
    input_len="Average prompt length (input tokens) per request.",
    output_len="Average generation length (output tokens) per request.",
    num_requests="Total number of requests to simulate.",
    max_num_batched_tokens="Per-step batched-token budget for the scheduler.",
)
def simulate_serving(
    model: str,
    gpu: str,
    concurrency: int,
    input_len: int,
    output_len: int,
    num_requests: int,
    max_num_batched_tokens: int,
) -> str:
    if model not in PRESET_MODELS:
        return f"ERROR: unknown model {model!r}. Available: {', '.join(sorted(PRESET_MODELS))}"
    if gpu not in PRESET_GPUS:
        return f"ERROR: unknown gpu {gpu!r}. Available: {', '.join(sorted(PRESET_GPUS))}"
    result = run_simulation(
        model_name=model,
        gpu_name=gpu,
        concurrency=concurrency,
        input_len=input_len,
        output_len=output_len,
        n_requests=num_requests,
        max_batched_tokens=max_num_batched_tokens,
    )
    return render_report(summarize_run(result),
                         cost_per_hour=PRESET_GPUS[gpu].cost_per_hour)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Configure hardware and workload for serving simulation."
    )
    parser.add_argument("--gpu", type=str, default="h100-sxm")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--num-concurrency", type=int, default=256)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--random-range-ratio", type=float, default=0.1)
    parser.add_argument("--num-requests", type=int, default=12800)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--bench-file", type=str, default=None,
                        help="Microbenchmark results JSON (optional).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    microbench = None
    if args.bench_file:
        with open(args.bench_file) as f:
            microbench = json.load(f)

    print("running workload simulation...")
    result = run_simulation(
        model_name=args.model,
        gpu_name=args.gpu,
        concurrency=args.num_concurrency,
        input_len=args.input_len,
        output_len=args.output_len,
        n_requests=args.num_requests,
        max_batched_tokens=args.max_num_batched_tokens,
        jitter=args.random_range_ratio,
        microbench=microbench,
    )
    print(render_report(summarize_run(result),
                        cost_per_hour=PRESET_GPUS[args.gpu].cost_per_hour))
