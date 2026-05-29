"""Hardware Pareto sweep — Stage 2 of the deployment-planning workflow.

For each candidate GPU: check the model weights fit on a single GPU,
run the serving simulator (which derives the KV-cache budget from the
remaining VRAM and admits requests against it), compute $/Mtoken, and
identify which candidates sit on the cost-vs-latency Pareto frontier
(no other candidate is both cheaper AND faster).

Concurrency is a *result* of the simulator's admission control at the
given ``request_rate``, not an input — so we don't pre-reject candidates
on a static worst-case KV calculation. If a candidate's KV budget is
too small to sustain the rate, the simulator saturates KV-bound and we
flag that in the table.

Assumes **single-GPU** deployment — the modeling tools don't model
TP/PP/DP scaling.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..base import tool
from ..modeling.configs.hw_specs import PRESET_GPUS
from ..modeling.configs.model_specs import PRESET_MODELS
from ..modeling.memory import weights_vram_gib
from ..modeling.report import ReportBuilder
from ..modeling.serving import run_simulation, summarize_run


_VRAM_HEADROOM = 0.90   # leave 10% of HBM for activations / framework overhead


@dataclass
class _CandidateRow:
    gpu: str
    fits_weights: bool             # model weights load on a single GPU
    weights_gib: float
    vram_cap_gib: float
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


def _evaluate(
    model: str,
    gpu_key: str,
    request_rate: float,
    input_len: int,
    output_len: int,
    num_requests: int,
    max_num_batched_tokens: int,
    max_concurrent_requests: int,
    range_ratio: float,
    target_latency: float | None,
) -> _CandidateRow:
    gpu = PRESET_GPUS[gpu_key]
    model_spec = PRESET_MODELS[model]

    # Weights-only fit gate: the model has to load before anything else
    # matters. KV admission is the simulator's job — it derives the KV
    # budget from VRAM minus weights minus framework overhead, and admits
    # requests subject to that budget.
    weights_g = weights_vram_gib(model_spec)
    vram_cap = gpu.mem_capacity / (1024 ** 3) * _VRAM_HEADROOM

    if weights_g > vram_cap:
        return _CandidateRow(
            gpu=gpu_key, fits_weights=False,
            weights_gib=weights_g, vram_cap_gib=vram_cap,
            kv_budget_gib=None, peak_kv_use_gib=None, kv_bound=False,
            request_latency_s=None, ttft_ms=None, tpot_ms=None,
            output_throughput_tps=None,
            cost_per_hour=gpu.cost_per_hour, cost_per_mtok=None,
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
        jitter=range_ratio,
    )
    s = summarize_run(result)
    if s is None or s.served_rate <= 0:
        return _CandidateRow(
            gpu=gpu_key, fits_weights=True,
            weights_gib=weights_g, vram_cap_gib=vram_cap,
            kv_budget_gib=None, peak_kv_use_gib=None, kv_bound=False,
            request_latency_s=None, ttft_ms=None, tpot_ms=None,
            output_throughput_tps=None,
            cost_per_hour=gpu.cost_per_hour, cost_per_mtok=None,
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
        (gpu.cost_per_hour / 3600.0) * 1e6 / output_throughput_tps
        if gpu.cost_per_hour > 0 and output_throughput_tps > 0 else None
    )
    meets = (request_latency_s <= target_latency) if target_latency is not None else None
    note = f"saturated: {s.saturation_reason}" if s.saturated else ""
    return _CandidateRow(
        gpu=gpu_key, fits_weights=True,
        weights_gib=weights_g, vram_cap_gib=vram_cap,
        kv_budget_gib=s.kv_budget_gib,
        peak_kv_use_gib=s.peak_kv_use_gib,
        kv_bound=kv_bound,
        request_latency_s=request_latency_s,
        ttft_ms=s.ttft_ms.mean,
        tpot_ms=s.tpot_ms.mean,
        output_throughput_tps=output_throughput_tps,
        cost_per_hour=gpu.cost_per_hour,
        cost_per_mtok=cost_per_mtok,
        meets_target=meets,
        saturated=s.saturated,
        note=note,
    )


def _pareto_mark(rows: list[_CandidateRow]) -> set[str]:
    """Names of candidates on the cost-vs-latency Pareto frontier.

    Only considers evaluated candidates with both cost and latency
    available; minimises both. A point is dominated if another point is
    no worse on both axes AND strictly better on at least one.
    """
    eligible = [r for r in rows
                if r.is_evaluated and r.cost_per_mtok is not None
                and not r.saturated]
    on_pareto: set[str] = set()
    for r in eligible:
        dominated = False
        for other in eligible:
            if other.gpu == r.gpu:
                continue
            cheaper_or_equal = other.cost_per_mtok <= r.cost_per_mtok
            faster_or_equal = other.request_latency_s <= r.request_latency_s
            strictly_better = (other.cost_per_mtok < r.cost_per_mtok
                               or other.request_latency_s < r.request_latency_s)
            if cheaper_or_equal and faster_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            on_pareto.add(r.gpu)
    return on_pareto


def _format_table(
    rows: list[_CandidateRow],
    on_pareto: set[str],
    target_latency: float | None,
) -> str:
    rb = ReportBuilder(width=110)
    rb.banner("HARDWARE PARETO SWEEP")
    rb.line()
    headers = ["gpu", "fits?", "KV peak", "req lat (s)", "ttft (ms)",
               "tpot (ms)", "$/1M tok", "meets target", "pareto"]
    table_rows = []
    for r in rows:
        if not r.fits_weights:
            table_rows.append([r.gpu, "no", "—", "—", "—", "—", "—", "—", "—"])
            continue
        if not r.is_evaluated:
            table_rows.append([r.gpu, "yes", "—", "—", "—", "—", "—", "—", "—"])
            continue
        fits_str = "KV-bound" if r.kv_bound else "yes"
        kv_str = f"{r.kv_peak_pct:.0f}%" if r.kv_peak_pct is not None else "—"
        meets = "—" if r.meets_target is None else ("✓" if r.meets_target else "✗")
        cost_str = f"${r.cost_per_mtok:.3f}" if r.cost_per_mtok is not None else "—"
        pareto = "★" if r.gpu in on_pareto else " "
        ttft_str = f"{r.ttft_ms:.0f}" if r.ttft_ms is not None else "—"
        tpot_str = f"{r.tpot_ms:.2f}" if r.tpot_ms is not None else "—"
        table_rows.append([
            r.gpu, fits_str, kv_str,
            f"{r.request_latency_s:.3f}",
            ttft_str, tpot_str,
            cost_str,
            meets,
            pareto,
        ])
    rb.table(
        headers=headers, rows=table_rows,
        col_widths=[12, 9, 8, 12, 10, 10, 10, 13, 7],
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
    "Stage 2 of the deployment-planning workflow: evaluate a list of "
    "single-GPU hardware candidates against a workload and return a "
    "cost-vs-latency Pareto table. For each candidate it checks that the "
    "model weights fit on a single GPU, runs the serving simulator (the "
    "simulator derives the KV-cache budget from VRAM minus weights minus "
    "framework overhead and admits requests against it), and computes $/1M "
    "output tokens. Candidates whose KV budget is too small to sustain the "
    "rate saturate KV-bound and are flagged accordingly. All workload knobs "
    "come from the YAML; you only pass the candidate hardware list.",
    workload_file="Workspace-relative path to a WorkloadProfile YAML "
                  "(e.g. 'stages/01_workload.yaml'). Must contain `model`, "
                  "`request_rate`, `input_len`, `output_len`, `num_requests`, "
                  "`max_num_batched_tokens`. `max_concurrent_requests` and "
                  "`target_request_latency_s` are optional.",
    candidates="List of GPU preset names (PRESET_GPUS keys). Ask the user "
               "to confirm the candidate set; don't silently sweep all GPUs.",
)
def pareto_sweep(
    workload_file: str,
    candidates: list,
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
        return ("ERROR: no candidates provided. Ask the user which GPUs to "
                "evaluate.")
    unknown = [c for c in candidates if c not in PRESET_GPUS]
    if unknown:
        return (f"ERROR: unknown gpu(s) {unknown}. "
                f"Available: {', '.join(sorted(PRESET_GPUS))}")
    if request_rate <= 0:
        return "ERROR: request_rate must be > 0."

    target = target_request_latency_s if target_request_latency_s > 0 else None
    rows = [
        _evaluate(model, c, request_rate, input_len, output_len,
                  num_requests, max_num_batched_tokens,
                  max_concurrent_requests, range_ratio, target)
        for c in candidates
    ]
    on_pareto = _pareto_mark(rows)
    return _format_table(rows, on_pareto, target)
