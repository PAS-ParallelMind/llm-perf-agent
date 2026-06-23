"""Hardware Pareto sweep — Stage 2 of the deployment-planning workflow.

For each candidate (gpu, tp, dp): check the model weights fit on the
replica's aggregate VRAM (TP shards weights across `tp` GPUs), run the
serving simulator (which derives the KV-cache budget from the remaining
per-replica VRAM and admits requests against it), compute $/Mtoken, and
identify which candidates sit on the cost-vs-latency Pareto frontier
(no other candidate is both cheaper AND faster).

Concurrency is a *result* of the simulator's admission control at the
given ``request_rate``, not an input — so we don't pre-reject candidates
on a static worst-case KV calculation. If a candidate's KV budget is
too small to sustain the rate, the simulator saturates KV-bound and we
flag that in the table.

The sweep runs in one ``latency_source`` mode per call — ``baseline``
(microbench-calibrated realistic projection, default) or
``theoretical`` (analytic peak ceiling). Stage 2 of the planning
workflow invokes this tool twice, once per mode, and presents both
tables.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..base import tool
from ..modeling.configs.hw_specs import PRESET_GPUS
from ..modeling.configs.model_specs import PRESET_MODELS
from ..modeling.latency import MODES, ParallelConfig
from ..modeling.memory import weights_vram_gib
from ..modeling.report import ReportBuilder
from ..modeling.serving import run_simulation, summarize_run


_VRAM_HEADROOM = 0.90   # leave 10% of HBM for activations / framework overhead


@dataclass
class _CandidateRow:
    gpu: str
    tp: int
    dp: int
    fits_weights: bool             # weights fit on a replica (tp * per-gpu VRAM)
    weights_gib: float
    vram_cap_gib: float            # per-replica usable VRAM (tp * mem * headroom)
    # Simulator-derived KV stats (None if the simulator never ran).
    kv_budget_gib: float | None
    peak_kv_use_gib: float | None
    kv_bound: bool                 # simulator saturated for KV reasons
    request_latency_s: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    # output_throughput_tps is kept for the $/Mtok computation but not
    # surfaced in the table (at sub-saturation it just mirrors offered load).
    output_throughput_tps: float | None
    cost_per_hour: float
    cost_per_mtok: float | None    # USD per 1M output tokens
    meets_target: bool | None
    saturated: bool = False        # diverging latency; excluded from frontier
    note: str = ""

    @property
    def is_evaluated(self) -> bool:
        return self.fits_weights and self.request_latency_s is not None

    @property
    def kv_peak_pct(self) -> float | None:
        if self.kv_budget_gib and self.peak_kv_use_gib is not None and self.kv_budget_gib > 0:
            return self.peak_kv_use_gib / self.kv_budget_gib * 100
        return None


def _enumerate_parallelism(model_spec, count: int) -> list[ParallelConfig]:
    """All valid (tp, dp) splits with tp*dp <= count and tp dividing
    the head counts. Single-GPU (tp=1, dp=1) is always included."""
    out: list[ParallelConfig] = []
    head_lcd = min(model_spec.n_attention_heads, model_spec.n_kv_heads)
    for tp in range(1, count + 1):
        if head_lcd % tp != 0:
            continue
        for dp in range(1, count // tp + 1):
            out.append(ParallelConfig(tp=tp, dp=dp))
    return out


def _evaluate(
    model: str,
    gpu_key: str,
    parallel: ParallelConfig,
    request_rate: float,
    input_len: int,
    output_len: int,
    num_requests: int,
    max_num_batched_tokens: int,
    max_concurrent_requests: int,
    range_ratio: float,
    target_latency: float | None,
    latency_source: str,
) -> _CandidateRow:
    gpu = PRESET_GPUS[gpu_key]
    model_spec = PRESET_MODELS[model]
    deployment_cost_per_hour = gpu.cost_per_hour * parallel.n_gpus

    # Weights-only fit gate: TP shards weights across `tp` GPUs, so the
    # replica's aggregate VRAM is `tp * per_gpu_mem`. KV admission is the
    # simulator's job — it derives the KV budget from per-replica VRAM
    # minus weights minus framework overhead, and admits requests
    # against it.
    weights_g = weights_vram_gib(model_spec)
    vram_cap = gpu.mem_capacity / (1024 ** 3) * _VRAM_HEADROOM * parallel.tp

    if weights_g > vram_cap:
        return _CandidateRow(
            gpu=gpu_key, tp=parallel.tp, dp=parallel.dp, fits_weights=False,
            weights_gib=weights_g, vram_cap_gib=vram_cap,
            kv_budget_gib=None, peak_kv_use_gib=None, kv_bound=False,
            request_latency_s=None, ttft_ms=None, tpot_ms=None,
            output_throughput_tps=None,
            cost_per_hour=deployment_cost_per_hour, cost_per_mtok=None,
            meets_target=None,
            note=f"weights don't fit: {weights_g:.1f} > {vram_cap:.1f} GiB usable",
        )

    result = run_simulation(
        model_name=model, gpu_name=gpu_key,
        request_rate=request_rate,
        input_len=input_len, output_len=output_len,
        n_requests=num_requests,
        max_batched_tokens=max_num_batched_tokens,
        max_concurrent_requests=max_concurrent_requests,
        parallel=parallel,
        jitter=range_ratio,
        latency_source=latency_source,
    )
    s = summarize_run(result)
    if s is None or s.served_rate <= 0:
        return _CandidateRow(
            gpu=gpu_key, tp=parallel.tp, dp=parallel.dp, fits_weights=True,
            weights_gib=weights_g, vram_cap_gib=vram_cap,
            kv_budget_gib=None, peak_kv_use_gib=None, kv_bound=False,
            request_latency_s=None, ttft_ms=None, tpot_ms=None,
            output_throughput_tps=None,
            cost_per_hour=deployment_cost_per_hour, cost_per_mtok=None,
            meets_target=None,
            note="simulation produced no usable result",
        )

    # KV-bound saturation = the rate exhausted the KV budget. Treat this
    # as "doesn't fit at this rate" in the report — the candidate can't
    # admit enough concurrent requests to sustain the offered load.
    kv_bound = s.saturated and "KV-cache" in s.saturation_reason
    output_throughput_tps = s.served_rate * output_len
    request_latency_s = s.e2e_s.mean
    cost_per_mtok = (
        (deployment_cost_per_hour / 3600.0) * 1e6 / output_throughput_tps
        if deployment_cost_per_hour > 0 and output_throughput_tps > 0 else None
    )
    meets = (request_latency_s <= target_latency) if target_latency is not None else None
    note = f"saturated: {s.saturation_reason}" if s.saturated else ""
    return _CandidateRow(
        gpu=gpu_key, tp=parallel.tp, dp=parallel.dp, fits_weights=True,
        weights_gib=weights_g, vram_cap_gib=vram_cap,
        kv_budget_gib=s.kv_budget_gib,
        peak_kv_use_gib=s.peak_kv_use_gib,
        kv_bound=kv_bound,
        request_latency_s=request_latency_s,
        ttft_ms=s.ttft_ms.mean,
        tpot_ms=s.tpot_ms.mean,
        output_throughput_tps=output_throughput_tps,
        cost_per_hour=deployment_cost_per_hour,
        cost_per_mtok=cost_per_mtok,
        meets_target=meets,
        saturated=s.saturated,
        note=note,
    )


def _row_key(r: _CandidateRow) -> tuple[str, int, int]:
    return (r.gpu, r.tp, r.dp)


def _pareto_mark(rows: list[_CandidateRow]) -> set[tuple[str, int, int]]:
    """(gpu, tp, dp) keys on the cost-vs-latency Pareto frontier.

    Only considers evaluated candidates with both cost and latency
    available; minimises both. A point is dominated if another point is
    no worse on both axes AND strictly better on at least one.
    """
    eligible = [r for r in rows
                if r.is_evaluated and r.cost_per_mtok is not None
                and not r.saturated]
    on_pareto: set[tuple[str, int, int]] = set()
    for r in eligible:
        dominated = False
        rk = _row_key(r)
        for other in eligible:
            if _row_key(other) == rk:
                continue
            cheaper_or_equal = other.cost_per_mtok <= r.cost_per_mtok
            faster_or_equal = other.request_latency_s <= r.request_latency_s
            strictly_better = (other.cost_per_mtok < r.cost_per_mtok
                               or other.request_latency_s < r.request_latency_s)
            if cheaper_or_equal and faster_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            on_pareto.add(rk)
    return on_pareto


def _format_table(
    rows: list[_CandidateRow],
    on_pareto: set[tuple[str, int, int]],
    target_latency: float | None,
    latency_source: str,
) -> str:
    rb = ReportBuilder(width=118)
    rb.banner(f"HARDWARE PARETO SWEEP — {latency_source.upper()} MODE")
    rb.line()
    headers = ["gpu", "tp", "dp", "n_gpu", "fits?", "KV peak", "req lat (s)",
               "ttft (ms)", "tpot (ms)", "$/1M tok", "meets target", "pareto"]
    table_rows = []
    for r in rows:
        n_gpu = r.tp * r.dp
        if not r.fits_weights:
            table_rows.append([r.gpu, str(r.tp), str(r.dp), str(n_gpu),
                               "no", "—", "—", "—", "—", "—", "—", "—"])
            continue
        if not r.is_evaluated:
            table_rows.append([r.gpu, str(r.tp), str(r.dp), str(n_gpu),
                               "yes", "—", "—", "—", "—", "—", "—", "—"])
            continue
        fits_str = "KV-bound" if r.kv_bound else "yes"
        kv_str = f"{r.kv_peak_pct:.0f}%" if r.kv_peak_pct is not None else "—"
        meets = "—" if r.meets_target is None else ("✓" if r.meets_target else "✗")
        cost_str = f"${r.cost_per_mtok:.3f}" if r.cost_per_mtok is not None else "—"
        pareto = "★" if _row_key(r) in on_pareto else " "
        ttft_str = f"{r.ttft_ms:.0f}" if r.ttft_ms is not None else "—"
        tpot_str = f"{r.tpot_ms:.2f}" if r.tpot_ms is not None else "—"
        table_rows.append([
            r.gpu, str(r.tp), str(r.dp), str(n_gpu),
            fits_str, kv_str,
            f"{r.request_latency_s:.3f}",
            ttft_str, tpot_str,
            cost_str,
            meets,
            pareto,
        ])
    rb.table(
        headers=headers, rows=table_rows,
        col_widths=[12, 4, 4, 6, 9, 8, 12, 10, 10, 10, 13, 7],
    )
    rb.line()
    for r in rows:
        if r.note:
            rb.line(f"  {r.gpu}: {r.note}")
    if target_latency is not None:
        rb.line(f"target_request_latency_s = {target_latency} "
                f"(✓ = meets, ✗ = misses)")
    rb.line("'KV peak' = peak KV-cache used as a % of the candidate's KV "
            "budget (VRAM after weights + framework overhead). 'KV-bound' "
            "in `fits?` means the rate exhausted the budget — concurrency "
            "couldn't grow enough to sustain the offered load.")
    rb.line("★ = on the cost-vs-latency Pareto frontier "
            "(no other candidate is both cheaper AND faster).")
    rb.line("Costs are owned-hardware TCO (MSRP amortised + electricity) — "
            "see list_gpus for the derivation. Cloud rental runs higher.")
    rb.rule("=")
    return rb.build()


@tool(
    "Stage 2 of the deployment-planning workflow: for each GPU type the "
    "user has, enumerate every valid (tp, dp) split that fits within the "
    "available count, run the serving simulator on each, and return a "
    "cost-vs-latency Pareto table. The tool runs in ONE `latency_source` "
    "mode per call — call it twice (once 'baseline', once 'theoretical') "
    "to produce both the realistic-projection table and the analytic-"
    "ceiling table. Cost in `$/1M tok` is the deployment cost (n_gpus × "
    "per-GPU $/h), so heavier configurations are correctly penalised. "
    "All workload knobs come from the YAML.",
    workload_file="Workspace-relative path to a WorkloadProfile YAML "
                  "(e.g. 'stages/01_workload.yaml'). Must contain `model`, "
                  "`request_rate`, `input_len`, `output_len`, `num_requests`, "
                  "`max_num_batched_tokens`. `max_concurrent_requests` and "
                  "`target_request_latency_s` are optional.",
    candidates="List of `{gpu: <PRESET_GPUS key>, count: <int>}` entries "
               "describing what the user has available. The tool enumerates "
               "every (tp, dp) with tp*dp <= count and tp dividing the "
               "model's KV-head count.",
    latency_source="'baseline' (default) — microbench-calibrated realistic "
                   "projection; this is what the deployment will actually "
                   "deliver. 'theoretical' — analytic peak ceiling "
                   "(efficiency=1.0 every op); useful for headroom analysis "
                   "but over-predicts throughput on real hardware.",
)
def pareto_sweep(
    workload_file: str,
    candidates: list,
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
    target_request_latency_s = wf.get("target_request_latency_s", 0.0)

    if model not in PRESET_MODELS:
        return (f"ERROR: unknown model {model!r}. "
                f"Available: {', '.join(sorted(PRESET_MODELS))}")
    if not candidates:
        return ("ERROR: no candidates provided. Ask the user what GPUs they "
                "have available, e.g. [{'gpu': 'h200-nvl', 'count': 8}].")
    if latency_source not in MODES:
        return f"ERROR: latency_source={latency_source!r} must be one of {MODES}."

    # Validate each entry has gpu + count; collect any errors up front so
    # the user can fix them in one round trip instead of N.
    parsed: list[tuple[str, int]] = []
    for i, c in enumerate(candidates):
        if not isinstance(c, dict) or "gpu" not in c or "count" not in c:
            return (f"ERROR: candidates[{i}]={c!r} — each entry must be "
                    f"{{gpu: str, count: int}}.")
        gpu_key, count = c["gpu"], c["count"]
        if gpu_key not in PRESET_GPUS:
            return (f"ERROR: candidates[{i}].gpu={gpu_key!r} is not a "
                    f"PRESET_GPUS key. Available: "
                    f"{', '.join(sorted(PRESET_GPUS))}")
        if not isinstance(count, int) or count < 1:
            return (f"ERROR: candidates[{i}].count={count!r} must be a "
                    f"positive int.")
        parsed.append((gpu_key, count))
    if request_rate <= 0:
        return "ERROR: request_rate must be > 0."

    model_spec = PRESET_MODELS[model]
    target = target_request_latency_s if target_request_latency_s > 0 else None
    rows: list[_CandidateRow] = []
    for gpu_key, count in parsed:
        for parallel in _enumerate_parallelism(model_spec, count):
            rows.append(_evaluate(
                model, gpu_key, parallel, request_rate, input_len, output_len,
                num_requests, max_num_batched_tokens, max_concurrent_requests,
                range_ratio, target, latency_source,
            ))
    on_pareto = _pareto_mark(rows)
    return _format_table(rows, on_pareto, target, latency_source)
