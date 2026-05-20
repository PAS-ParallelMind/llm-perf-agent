"""GPU / model catalog tools.

Expose the ``PRESET_GPUS`` and ``PRESET_MODELS`` tables so the agent can
*discover* which presets exist and *look up* their specs — rather than
guessing preset names (and hitting "unknown gpu" errors) or reasoning
about hardware capability from memory.
"""
from __future__ import annotations

from ..base import tool
from .configs.hw_specs import PRESET_GPUS
from .configs.model_specs import PRESET_MODELS
from .report import ReportBuilder


def _tflops(flops: float) -> str:
    return f"{flops / 1e12:.0f}" if flops else "—"


@tool(
    "List the GPU presets available to the modeling tools, with compute "
    "throughput (BF16 / FP8 TFLOP/s), HBM bandwidth, and memory capacity. "
    "Call this when the user mentions a GPU — to find the exact `gpu` "
    "preset name to pass to the other tools and to look up its specs. If "
    "the user's GPU is not listed, say so and pick the closest match or "
    "ask to add it as a preset."
)
def list_gpus() -> str:
    rb = ReportBuilder(width=72)
    rb.banner("AVAILABLE GPU PRESETS")
    rows = []
    for key, g in PRESET_GPUS.items():
        rows.append([
            key,
            _tflops(g.bf16_flops),
            _tflops(g.fp8_flops),
            f"{g.mem_bandwidth / 1e12:.2f}",
            f"{g.mem_capacity / 1e9:.0f}",
        ])
    rb.table(
        headers=["preset", "BF16 TF/s", "FP8 TF/s", "HBM TB/s", "VRAM GB"],
        rows=rows,
        col_widths=[12, 10, 10, 9, 8],
    )
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
    rb = ReportBuilder(width=96)
    rb.banner("AVAILABLE MODEL PRESETS")
    rows = []
    for key, m in PRESET_MODELS.items():
        experts = f"{m.n_experts}/{m.top_k}" if m.n_experts else "dense"
        rows.append([
            key,
            m.ffn_weight_dtype,
            str(m.n_layers),
            str(m.hidden_size),
            experts,
            f"{m.max_seq_len:,}",
        ])
    rb.table(
        headers=["preset", "ffn dtype", "layers", "hidden", "experts/top_k", "max_seq_len"],
        rows=rows,
        col_widths=[46, 10, 7, 8, 14, 12],
    )
    rb.rule("=")
    return rb.build()
