"""VRAM estimation: model weights + KV cache.

Exposed as the ``estimate_memory`` tool; also runnable as a CLI for ad-hoc
sanity checks.
"""
from __future__ import annotations

import argparse

from ..base import tool
from .configs.model_specs import PRESET_MODELS, ModelConfig, get_quantization_bytes
from .report import ReportBuilder


_GIB = 1024 ** 3


def weights_vram_gib(model: ModelConfig) -> float:
    """Total weight VRAM in GiB."""
    orig_byte = get_quantization_bytes(model.model_orig_dtype)
    attn_byte = get_quantization_bytes(model.attn_weight_dtype)
    ffn_byte = get_quantization_bytes(model.ffn_weight_dtype)

    embedding_bytes = model.vocab_size * model.hidden_size * orig_byte

    # Attention projections (q, k, v, o) per layer.
    q_params = model.hidden_size * (model.n_attention_heads * model.head_dim)
    k_params = model.hidden_size * (model.n_kv_heads * model.head_dim)
    v_params = model.hidden_size * (model.n_kv_heads * model.head_dim)
    o_params = (model.n_attention_heads * model.head_dim) * model.hidden_size
    attn_bytes_per_layer = (q_params + k_params + v_params + o_params) * attn_byte

    # FFN: dense SwiGLU (3 matmuls) or MoE (router + per-expert SwiGLU).
    if model.n_experts > 0:
        router_bytes = model.hidden_size * model.n_experts * orig_byte
        params_per_expert = 3 * model.hidden_size * model.moe_intermediate_size
        ffn_bytes_per_layer = router_bytes + (params_per_expert * model.n_experts * ffn_byte)
    else:
        ffn_params = 3 * model.hidden_size * model.intermediate_size
        ffn_bytes_per_layer = ffn_params * ffn_byte

    norm_bytes_per_layer = 2 * model.hidden_size * orig_byte
    transformer_bytes = model.n_layers * (
        attn_bytes_per_layer + ffn_bytes_per_layer + norm_bytes_per_layer
    )

    final_norm_bytes = model.hidden_size * orig_byte
    lm_head_bytes = model.vocab_size * model.hidden_size * orig_byte

    total_bytes = embedding_bytes + transformer_bytes + final_norm_bytes + lm_head_bytes
    return total_bytes / _GIB


def kv_cache_vram_gib(
    model: ModelConfig,
    concurrency: int,
    context_length: int,
) -> float:
    """KV cache VRAM in GiB, accounting for sliding-window layers."""
    requested_len = min(context_length, model.max_seq_len)

    # Sum tokens-cached-at-this-layer across all layers (SWA layers cap to the window).
    tokens_per_request_summed = sum(
        model.effective_kv_tokens(layer_idx, requested_len)
        for layer_idx in range(model.n_layers)
    )

    kv_byte = get_quantization_bytes(model.kv_cache_dtype)
    bytes_per_token_per_layer = 2 * model.n_kv_heads * model.head_dim * kv_byte
    total_bytes = concurrency * tokens_per_request_summed * bytes_per_token_per_layer

    return total_bytes / _GIB


def _render_report(
    model_name: str,
    concurrency: int,
    context_length: int,
    kv_dtype: str,
    kv_dtype_bytes: float,
    weights_gib: float,
    kv_cache_gib: float,
) -> str:
    total_gib = weights_gib + kv_cache_gib
    rb = ReportBuilder(width=55)
    rb.banner("LLM VRAM ESTIMATION REPORT").line()
    rb.heading("Deployment Parameters")
    rb.kv("Model Name", model_name)
    rb.kv("Target Concurrency", f"{concurrency} concurrent requests")
    rb.kv("Context Length", f"{context_length:,} tokens / request")
    rb.kv("KV Cache Precision", f"{kv_dtype} ({kv_dtype_bytes} bytes/param)")
    rb.line()
    rb.heading("Memory Footprint")
    rb.kv("Model Size", f"{weights_gib:>8.3f} GiB")
    rb.kv("KV Cache Size", f"{kv_cache_gib:>8.3f} GiB")
    rb.rule()
    rb.kv("Total Required VRAM", f"{total_gib:>8.3f} GiB")
    rb.rule("=")
    return rb.build()


@tool(
    "Estimate GPU memory required to serve an LLM (weights + KV cache). "
    "Returns a structured VRAM breakdown report.",
    model="Preset model name (e.g., 'openai/gpt-oss-20b', "
          "'Qwen/Qwen3-Coder-30B-A3B-Instruct'). Must exist in PRESET_MODELS.",
    concurrency="Number of concurrent in-flight requests sharing the KV cache.",
    context_length="Per-request context length in tokens (capped at the "
                   "model's max_seq_len).",
)
def estimate_memory(model: str, concurrency: int, context_length: int) -> str:
    if model not in PRESET_MODELS:
        return f"ERROR: unknown model {model!r}. Available: {', '.join(sorted(PRESET_MODELS))}"
    spec = PRESET_MODELS[model]
    weights_gib = weights_vram_gib(spec)
    kv_gib = kv_cache_vram_gib(spec, concurrency, context_length)
    return _render_report(
        model_name=model,
        concurrency=concurrency,
        context_length=context_length,
        kv_dtype=spec.kv_cache_dtype,
        kv_dtype_bytes=get_quantization_bytes(spec.kv_cache_dtype),
        weights_gib=weights_gib,
        kv_cache_gib=kv_gib,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate VRAM (weights + KV cache) for LLM deployment."
    )
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    if args.model not in PRESET_MODELS:
        print(f"\n[!] Error: model {args.model!r} not in PRESET_MODELS.")
        raise SystemExit(1)

    print(estimate_memory(
        model=args.model,
        concurrency=args.concurrency,
        context_length=args.context_length,
    ))
