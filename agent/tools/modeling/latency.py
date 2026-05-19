"""Analytical forward-pass latency model.

Computes per-operation compute / memory / roofline latencies for a single
transformer forward pass. Used standalone via the ``estimate_latency``
tool, and as the inner loop of the serving simulator.

Conventions:
* ``compute_s`` / ``memory_s`` / ``roofline_s`` are all in seconds.
* "Roofline" time defaults to ``max(compute, memory)`` (perfect overlap
  of one resource bound). A soft-roofline p-norm may be supplied via
  microbenchmark data to model partial overlap.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from typing import Any, Iterator

import numpy as np

from ..base import tool
from .configs.hw_specs import PRESET_GPUS, HardwareConfig
from .configs.model_specs import PRESET_MODELS, ModelConfig, get_quantization_bytes
from .report import ReportBuilder


# ---------------------------------------------------------------------------
# Per-operation latency primitives
# ---------------------------------------------------------------------------

@dataclass
class OperationLatency:
    """Compute / memory / roofline time for one operation, in seconds."""

    compute_s: float = 0.0
    memory_s: float = 0.0
    roofline_s: float = 0.0

    def apply_roofline(self, p_norm: float | None = None) -> None:
        """Set ``roofline_s`` from the current compute/memory components.

        ``p_norm=None`` → pure max (perfect overlap of the slower resource).
        ``p_norm=p``    → soft roofline: ``(c^p + m^p) ** (1/p)``.
        """
        if p_norm:
            self.roofline_s = (
                self.compute_s ** p_norm + self.memory_s ** p_norm
            ) ** (1.0 / p_norm)
        else:
            self.roofline_s = max(self.compute_s, self.memory_s)

    def __add__(self, other: "OperationLatency") -> "OperationLatency":
        if not isinstance(other, OperationLatency):
            return NotImplemented
        return OperationLatency(
            compute_s=self.compute_s + other.compute_s,
            memory_s=self.memory_s + other.memory_s,
            roofline_s=self.roofline_s + other.roofline_s,
        )

    def __iadd__(self, other: "OperationLatency") -> "OperationLatency":
        if not isinstance(other, OperationLatency):
            return NotImplemented
        self.compute_s += other.compute_s
        self.memory_s += other.memory_s
        self.roofline_s += other.roofline_s
        return self


@dataclass
class OpBreakdown:
    """Per-op latency breakdown of a single forward pass."""

    qkv_proj: OperationLatency = field(default_factory=OperationLatency)
    attn_core: OperationLatency = field(default_factory=OperationLatency)
    o_proj: OperationLatency = field(default_factory=OperationLatency)
    up_gate_proj: OperationLatency = field(default_factory=OperationLatency)
    down_proj: OperationLatency = field(default_factory=OperationLatency)
    lm_head: OperationLatency = field(default_factory=OperationLatency)

    def iter_ops(self) -> Iterator[tuple[str, OperationLatency]]:
        for f in fields(self):
            yield f.name, getattr(self, f.name)

    def total(self) -> OperationLatency:
        agg = OperationLatency()
        for _, op in self.iter_ops():
            agg += op
        return agg

    def accumulate(self, other: "OpBreakdown") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def attention_total(self) -> OperationLatency:
        return self.qkv_proj + self.attn_core + self.o_proj

    def ffn_total(self) -> OperationLatency:
        return self.up_gate_proj + self.down_proj


@dataclass
class Latency:
    """Per-request latency, split by phase."""

    waiting: OperationLatency = field(default_factory=OperationLatency)
    prefill: OpBreakdown = field(default_factory=OpBreakdown)
    decode: OpBreakdown = field(default_factory=OpBreakdown)


@dataclass
class Request:
    """One in-flight request tracked by the serving simulator."""

    id: Any
    arrival_s: float
    prompt_tokens: int
    gen_tokens: int
    kv_tokens: int = 0           # KV cached before the current step
    tokens_this_step: int = 0    # how many tokens this request runs THIS step
    finish_s: float = 0.0
    latency: Latency = field(default_factory=Latency)


def total_tokens_in_batch(requests: list[Request]) -> int:
    return sum(r.tokens_this_step for r in requests)


# ---------------------------------------------------------------------------
# Roofline lookup helpers
# ---------------------------------------------------------------------------

def achievable_bandwidth(data_size_bytes: float, mem_bw_curve: dict) -> float:
    """Interpolate the achievable HBM bandwidth (bytes/s) for a given working set.

    ``mem_bw_curve`` maps ``{size_in_MiB (str|int): bandwidth_in_GBs (float)}``;
    the curve is sampled in log2 space because microbenchmarks grow
    exponentially and bandwidth scales logarithmically with cache boundaries.
    """
    curve = {int(k): v for k, v in mem_bw_curve.items()}
    data_size_mib = data_size_bytes / (1024 ** 2)
    points = sorted(curve.items())
    x_mib = np.array([p[0] for p in points])
    y_gbs = np.array([p[1] for p in points])
    estimated_gbs = np.interp(np.log2(data_size_mib), np.log2(x_mib), y_gbs)
    return estimated_gbs * 1e9


def op_latency_from_roofline(
    total_flops: float,
    compute_dtype: str,
    total_bytes: float,
    microbench: dict | None,
    gpu: HardwareConfig,
) -> OperationLatency:
    """Compute the roofline latency for one op given its FLOP and byte counts."""
    # Today we only model bf16 compute throughput. FP8/FP4 paths exist on
    # the GPU spec but are gated until benchmark calibration lands.
    compute_flops = gpu.bf16_flops
    mem_bandwidth = gpu.mem_bandwidth
    p_norm = None
    if microbench:
        compute_flops = microbench["compute"][compute_dtype] * 1e12
        mem_bandwidth = achievable_bandwidth(total_bytes, microbench["memory"])
        p_norm = microbench["p"]

    op = OperationLatency(
        compute_s=total_flops / compute_flops,
        memory_s=total_bytes / mem_bandwidth,
    )
    op.apply_roofline(p_norm)
    return op


# ---------------------------------------------------------------------------
# Matmul / attention primitives
# ---------------------------------------------------------------------------

def matmul_latency(
    n_tokens: int,
    in_dim: int,
    out_dim: int,
    microbench: dict | None,
    gpu: HardwareConfig,
    weight_dtype: str,
    act_dtype: str,
) -> OperationLatency:
    """Roofline for a single GEMM of shape (n_tokens, in_dim) @ (in_dim, out_dim)."""
    weight_byte = get_quantization_bytes(weight_dtype)
    act_byte = get_quantization_bytes(act_dtype)

    total_flops = 2 * n_tokens * in_dim * out_dim
    total_bytes = act_byte * (n_tokens * in_dim + n_tokens * out_dim) + weight_byte * in_dim * out_dim
    return op_latency_from_roofline(total_flops, act_dtype, total_bytes, microbench, gpu)


def grouped_matmul_latency(
    shapes: list[tuple[int, int, int]],
    microbench: dict | None,
    gpu: HardwareConfig,
    weight_dtype: str,
    act_dtype: str,
) -> OperationLatency:
    """Roofline for grouped GEMM (MoE) — sum FLOPs / bytes across non-empty groups."""
    weight_byte = get_quantization_bytes(weight_dtype)
    act_byte = get_quantization_bytes(act_dtype)

    total_flops = 0
    total_bytes = 0
    for n_tokens, in_dim, out_dim in shapes:
        if n_tokens == 0:
            continue
        total_flops += 2 * n_tokens * in_dim * out_dim
        total_bytes += act_byte * (n_tokens * in_dim + n_tokens * out_dim) + weight_byte * in_dim * out_dim

    return op_latency_from_roofline(total_flops, act_dtype, total_bytes, microbench, gpu)


def attention_core_latency(
    requests: list[Request],
    q_proj_dim: int,        # head_dim * n_attention_heads
    gqa_group: int,         # n_attention_heads / n_kv_heads
    window: int,            # effective attention window (sliding cap or max_seq_len)
    microbench: dict | None,
    gpu: HardwareConfig,
    act_dtype: str,
) -> OperationLatency:
    """Roofline for Q@K^T followed by Score@V, under causal + sliding-window masking."""
    act_byte = get_quantization_bytes(act_dtype)

    total_flops = 0
    total_bytes = 0

    for r in requests:
        n_tokens = r.tokens_this_step
        kv_prior = r.kv_tokens

        # Causal+SWA dot-product count. Each query at chunk-position i in
        # [0, n_tokens) attends to min(window, kv_prior + i + 1) keys.
        # Closed-form so we stay O(1) for very long prefills.
        if window >= kv_prior + n_tokens:
            # No sliding cap kicks in: causal triangle over the prior KV.
            causal_pairs = n_tokens * kv_prior + n_tokens * (n_tokens + 1) // 2
        elif window <= kv_prior:
            # Every query is already past the window; each sees exactly W keys.
            causal_pairs = n_tokens * window
        else:
            # Mixed: queries 0..i_star are uncapped (causal), the rest clipped.
            i_star = window - kv_prior - 1
            uncapped = (i_star + 1) * kv_prior + (i_star + 1) * (i_star + 2) // 2
            capped = (n_tokens - i_star - 1) * window
            causal_pairs = uncapped + capped

        # Q@K^T and Score@V each contribute 2 * causal_pairs * q_proj_dim FLOPs.
        total_flops += 4 * causal_pairs * q_proj_dim

        # Bytes: union of unique keys/values read across this chunk.
        kv_unique = n_tokens + window if kv_prior + n_tokens > window else kv_prior + n_tokens
        bytes_q = n_tokens * q_proj_dim
        bytes_kv = 2 * kv_unique * q_proj_dim // gqa_group
        bytes_o = n_tokens * q_proj_dim
        total_bytes += act_byte * (bytes_q + bytes_kv + bytes_o)

    return op_latency_from_roofline(total_flops, act_dtype, total_bytes, microbench, gpu)


# ---------------------------------------------------------------------------
# Layer / full-forward composition
# ---------------------------------------------------------------------------

def transformer_layer_latency(
    requests: list[Request],
    layer_idx: int,
    microbench: dict | None,
    gpu: HardwareConfig,
    model: ModelConfig,
    expert_cache: dict | None = None,
) -> OpBreakdown:
    """Roofline for one transformer layer applied to ``requests``."""
    breakdown = OpBreakdown()
    n_tokens = total_tokens_in_batch(requests)

    # ----- GQA attention -----
    gqa_group = model.n_attention_heads // model.n_kv_heads

    breakdown.qkv_proj = matmul_latency(
        n_tokens,
        model.hidden_size,
        model.head_dim * (model.n_attention_heads + 2 * model.n_kv_heads),
        microbench, gpu,
        model.attn_weight_dtype, model.activation_dtype,
    )

    window = model.sliding_window if model.layer_uses_sliding_window(layer_idx) else model.max_seq_len
    breakdown.attn_core = attention_core_latency(
        requests,
        q_proj_dim=model.head_dim * model.n_attention_heads,
        gqa_group=gqa_group,
        window=window,
        microbench=microbench,
        gpu=gpu,
        act_dtype=model.activation_dtype,
    )

    breakdown.o_proj = matmul_latency(
        n_tokens,
        model.head_dim * model.n_attention_heads,
        model.hidden_size,
        microbench, gpu,
        model.attn_weight_dtype, model.activation_dtype,
    )

    # ----- FFN: dense or MoE -----
    if model.n_experts <= 1:
        # Dense SwiGLU: gate+up are typically fused into a 2*intermediate-wide GEMM.
        breakdown.up_gate_proj = matmul_latency(
            n_tokens, model.hidden_size, model.intermediate_size * 2,
            microbench, gpu, model.ffn_weight_dtype, model.activation_dtype,
        )
        breakdown.down_proj = matmul_latency(
            n_tokens, model.intermediate_size, model.hidden_size,
            microbench, gpu, model.ffn_weight_dtype, model.activation_dtype,
        )
    elif expert_cache is not None and n_tokens in expert_cache:
        cached = expert_cache[n_tokens]
        breakdown.up_gate_proj = cached["up_gate"]
        breakdown.down_proj = cached["down"]
    else:
        # MoE: simulate uniform-random expert selection to estimate
        # per-expert token loads, then grouped GEMM.
        token_counts = _sample_expert_loads(n_tokens, model.n_experts, model.top_k)

        up_gate_shapes = [
            (token_counts[e], model.hidden_size, model.moe_intermediate_size * 2)
            for e in range(model.n_experts)
        ]
        breakdown.up_gate_proj = grouped_matmul_latency(
            up_gate_shapes, microbench, gpu,
            model.ffn_weight_dtype, model.activation_dtype,
        )
        down_shapes = [
            (token_counts[e], model.moe_intermediate_size, model.hidden_size)
            for e in range(model.n_experts)
        ]
        breakdown.down_proj = grouped_matmul_latency(
            down_shapes, microbench, gpu,
            model.ffn_weight_dtype, model.activation_dtype,
        )
        if expert_cache is not None:
            expert_cache[n_tokens] = {
                "up_gate": breakdown.up_gate_proj,
                "down": breakdown.down_proj,
            }

    return breakdown


def _sample_expert_loads(n_tokens: int, n_experts: int, top_k: int) -> list[int]:
    """How many tokens each expert receives under uniform-random top-k routing."""
    counts = [0] * n_experts
    expert_ids = list(range(n_experts))
    for _ in range(n_tokens):
        for e in random.sample(expert_ids, top_k):
            counts[e] += 1
    return counts


def forward_pass_latency(
    requests: list[Request],
    microbench: dict | None,
    gpu: HardwareConfig,
    model: ModelConfig,
    expert_cache: dict | None = None,
) -> OpBreakdown:
    """Roofline for a full forward pass (all transformer layers + lm_head)."""
    total = OpBreakdown()
    for layer_idx in range(model.n_layers):
        total.accumulate(
            transformer_layer_latency(requests, layer_idx, microbench, gpu, model, expert_cache)
        )

    # lm_head runs on one token per request (the new query position).
    total.lm_head = matmul_latency(
        len(requests),
        model.hidden_size,
        model.vocab_size,
        microbench, gpu,
        model.model_orig_dtype, model.activation_dtype,
    )
    return total


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _bottleneck_label(op: OperationLatency) -> str:
    if op.compute_s > op.memory_s * 1.05:
        return "COMPUTE"
    if op.memory_s > op.compute_s * 1.05:
        return "MEMORY"
    return "BALANCED"


def _render_forward_report(
    model_name: str,
    gpu_name: str,
    batch_size: int,
    input_tokens: int,
    kv_cache_len: int,
    breakdown: OpBreakdown,
) -> str:
    rb = ReportBuilder(width=76)
    rb.banner("FORWARD-PASS LATENCY BREAKDOWN").line()
    rb.heading("Configuration")
    rb.kv("Model", model_name)
    rb.kv("GPU", gpu_name)
    rb.kv("Batch size", f"{batch_size} request(s)")
    rb.kv("Tokens / request", f"{input_tokens} (this forward pass)")
    rb.kv("KV cache / request", f"{kv_cache_len} tokens")
    rb.line()
    rb.heading("Per-Operation Latency")

    rows = []
    for name, op in breakdown.iter_ops():
        rows.append([
            name,
            f"{op.roofline_s * 1000:.3f}",
            f"{op.compute_s * 1000:.3f}",
            f"{op.memory_s * 1000:.3f}",
            _bottleneck_label(op),
        ])
    total = breakdown.total()
    rows.append([
        "TOTAL",
        f"{total.roofline_s * 1000:.3f}",
        f"{total.compute_s * 1000:.3f}",
        f"{total.memory_s * 1000:.3f}",
        _bottleneck_label(total),
    ])
    rb.table(
        headers=["Op", "Roofline(ms)", "Compute(ms)", "Memory(ms)", "Bottleneck"],
        rows=rows,
        col_widths=[14, 12, 11, 10, 10],
    )
    rb.rule("=")
    return rb.build()


# ---------------------------------------------------------------------------
# Tool: estimate_latency
# ---------------------------------------------------------------------------

@tool(
    "Estimate latency of a single transformer forward pass for a "
    "homogeneous batch. Returns a per-operation breakdown (compute vs. "
    "memory bound). For decode, set input_tokens=1 and kv_cache_len to "
    "the current context length; for prefill, set input_tokens to the "
    "prompt length and kv_cache_len=0.",
    model="Preset model name (must exist in PRESET_MODELS).",
    gpu="Preset GPU name (must exist in PRESET_GPUS).",
    batch_size="Number of requests in the batch.",
    input_tokens="Tokens per request being processed this forward pass "
                 "(1 = decode step, N = prefill chunk of N).",
    kv_cache_len="Existing KV cache length per request before this pass.",
)
def estimate_latency(
    model: str,
    gpu: str,
    batch_size: int,
    input_tokens: int,
    kv_cache_len: int,
) -> str:
    if model not in PRESET_MODELS:
        return f"ERROR: unknown model {model!r}. Available: {', '.join(sorted(PRESET_MODELS))}"
    if gpu not in PRESET_GPUS:
        return f"ERROR: unknown gpu {gpu!r}. Available: {', '.join(sorted(PRESET_GPUS))}"

    model_spec = PRESET_MODELS[model]
    gpu_spec = PRESET_GPUS[gpu]
    requests = [
        Request(
            id=i,
            arrival_s=0.0,
            prompt_tokens=input_tokens + kv_cache_len,
            gen_tokens=1,
            kv_tokens=kv_cache_len,
            tokens_this_step=input_tokens,
        )
        for i in range(batch_size)
    ]
    breakdown = forward_pass_latency(requests, None, gpu_spec, model_spec)
    return _render_forward_report(
        model_name=model,
        gpu_name=gpu,
        batch_size=batch_size,
        input_tokens=input_tokens,
        kv_cache_len=kv_cache_len,
        breakdown=breakdown,
    )
