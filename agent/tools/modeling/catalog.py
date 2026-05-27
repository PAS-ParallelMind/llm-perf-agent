"""GPU / model catalog tools.

Expose the ``PRESET_GPUS`` and ``PRESET_MODELS`` tables so the agent can
*discover* which presets exist and *look up* their specs — rather than
guessing preset names (and hitting "unknown gpu" errors) or reasoning
about hardware capability from memory.
"""
from __future__ import annotations

from ..base import tool
from .configs.hw_specs import (
    ELECTRICITY_USD_PER_KWH,
    PRESET_GPUS,
    PUE,
    USEFUL_LIFE_YEARS,
)
from .configs.model_specs import PRESET_MODELS
from .report import ReportBuilder


def _tflops(flops: float) -> str:
    return f"{flops / 1e12:.0f}" if flops else "—"


@tool(
    "List the GPU presets available to the modeling tools, with compute "
    "throughput (BF16 / FP8 TFLOP/s), HBM bandwidth, memory capacity, and "
    "owned-hardware $/GPU-hour derived from MSRP + TDP. Call this when "
    "the user mentions a GPU — to find the exact `gpu` preset name to "
    "pass to the other tools and to look up its specs. If the user's GPU "
    "is not listed, say so and pick the closest match or ask to add it "
    "as a preset."
)
def list_gpus() -> str:
    rb = ReportBuilder(width=92)
    rb.banner("AVAILABLE GPU PRESETS")
    rows = []
    for key, g in PRESET_GPUS.items():
        rows.append([
            key,
            _tflops(g.bf16_flops),
            _tflops(g.fp8_flops),
            f"{g.mem_bandwidth / 1e12:.2f}",
            f"{g.mem_capacity / 1e9:.0f}",
            f"${g.msrp_usd / 1000:.1f}k" if g.msrp_usd else "—",
            f"{g.tdp_watts:.0f}" if g.tdp_watts else "—",
            f"${g.cost_per_hour:.2f}" if g.cost_per_hour else "—",
        ])
    rb.table(
        headers=["preset", "BF16 TF/s", "FP8 TF/s", "HBM TB/s",
                 "VRAM GB", "MSRP", "TDP W", "$/hr (TCO)"],
        rows=rows,
        col_widths=[12, 10, 10, 9, 8, 7, 7, 11],
    )
    rb.line()
    rb.line(f"$/hr is OWNED-HARDWARE TCO = MSRP amortised over "
            f"{USEFUL_LIFE_YEARS:.0f} years + electricity "
            f"({PUE}× PUE × ${ELECTRICITY_USD_PER_KWH:.2f}/kWh). "
            f"Cloud RENTAL adds provider margin on top — typically 2-3× "
            f"higher. Adjust assumptions in hw_specs.py for your "
            f"electricity / depreciation policy.")
    rb.rule("=")
    return rb.build()


@tool(
    "List the model presets available to the modeling tools, with shape "
    "(layers, hidden size, experts) and weight dtype. Call this when the "
    "user mentions a model — to find the exact `model` preset name to pass "
    "to the other tools and to look up its architecture. If the user's "
    "model is not listed, say so and ask to add it as a preset."
)
def list_models() -> str:
    rb = ReportBuilder(width=112)
    rb.banner("AVAILABLE MODEL PRESETS")
    rows = []
    for key, m in PRESET_MODELS.items():
        experts = f"{m.n_experts}/{m.top_k}" if m.n_experts else "dense"
        weights = f"ffn={m.ffn_weight_dtype} attn={m.attn_weight_dtype}"
        rows.append([
            key,
            weights,
            str(m.n_layers),
            str(m.hidden_size),
            experts,
            f"{m.max_seq_len:,}",
            m.reasoning_mode,
        ])
    rb.table(
        headers=["preset", "weight dtypes", "layers", "hidden", "experts/top_k",
                 "max_seq_len", "reasoning"],
        rows=rows,
        col_widths=[46, 22, 7, 7, 14, 12, 9],
    )
    rb.line()
    rb.line("weight dtypes show the dtype the model is SHIPPED in. If ffn is "
            "already mxfp4 / int4 / fp8 / int8, the model is already quantized "
            "— don't suggest 're-quantizing' it (the user would have to find a "
            "different release).")
    rb.line("reasoning: none = no thinking tokens; hybrid = can toggle / tune "
            "effort (pick the served mode); always = always reasons.")
    rb.line("For hybrid/always, add the reasoning budget to output_len.")
    rb.rule("=")
    return rb.build()
