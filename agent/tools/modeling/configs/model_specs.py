"""Model preset specifications used by the modeling tools."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuantizationMethod(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"
    INT4 = "int4"
    FP4 = "fp4"
    MXFP4 = "mxfp4"


_BYTES_PER_PARAM: dict[str, float] = {
    QuantizationMethod.FP32:  4.0,
    QuantizationMethod.FP16:  2.0,
    QuantizationMethod.BF16:  2.0,
    QuantizationMethod.INT8:  1.0,
    QuantizationMethod.FP8:   1.0,
    QuantizationMethod.INT4:  0.5,
    QuantizationMethod.FP4:   0.5,
    QuantizationMethod.MXFP4: 0.5,
}


def get_quantization_bytes(method: str) -> float:
    """Bytes per stored parameter for a quantization method. Defaults to bf16 if unknown."""
    return _BYTES_PER_PARAM.get(method, 2.0)


@dataclass
class ModelConfig:
    """Transformer model shape + dtype configuration.

    Fields are grouped:

    * ``*_dtype``: precision used for weights / activations / KV cache.
    * Shape: layer count, hidden dims, head counts, FFN width.
    * MoE: number of experts, per-token top-k, per-expert FFN width.
    * Context: max sequence length, optional sliding-window size.
    """

    name: str

    # precisions
    model_orig_dtype: str   # embedding / lm_head / norms (and any non-quantized op)
    ffn_weight_dtype: str
    attn_weight_dtype: str
    activation_dtype: str
    kv_cache_dtype: str

    # transformer shape
    n_layers: int
    hidden_size: int
    vocab_size: int
    n_attention_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int

    # MoE
    n_experts: int
    top_k: int
    moe_intermediate_size: int

    # context length
    max_seq_len: int
    sliding_window: Optional[int] = None

    # Reasoning behaviour, which drives how much output (hidden chain-of-
    # thought + answer) a request generates:
    #   "none"   — never emits reasoning tokens (e.g. plain Instruct models)
    #   "hybrid" — can switch on/off or tune effort (e.g. Qwen3 think/no-think,
    #              gpt-oss reasoning effort); the served mode must be chosen
    #   "always" — always reasons (e.g. DeepSeek-R1 style)
    reasoning_mode: str = "none"

    def layer_uses_sliding_window(self, layer_idx: int) -> bool:
        """True if the given layer applies sliding-window attention.

        Today the only preset with ``sliding_window`` set is gpt-oss-20b,
        which interleaves SWA on even layers and full attention on odd
        layers. Other future models with ``sliding_window`` set will need
        their interleaving pattern encoded here.
        """
        if self.sliding_window is None:
            return False
        return layer_idx % 2 == 0

    def effective_kv_tokens(self, layer_idx: int, kv_tokens: int) -> int:
        """KV tokens stored at a layer (capped to ``sliding_window`` for SWA layers)."""
        if self.layer_uses_sliding_window(layer_idx):
            return min(kv_tokens, self.sliding_window)
        return kv_tokens


PRESET_MODELS: dict[str, ModelConfig] = {
    "openai/gpt-oss-20b": ModelConfig(
        name="openai/gpt-oss-20b",
        model_orig_dtype="bf16",
        ffn_weight_dtype="mxfp4",
        attn_weight_dtype="bf16",
        activation_dtype="bf16",
        kv_cache_dtype="bf16",
        n_layers=24,
        hidden_size=2880,
        vocab_size=201088,
        n_attention_heads=64,
        n_kv_heads=8,
        head_dim=64,
        intermediate_size=2880,
        moe_intermediate_size=2880,
        n_experts=32,
        top_k=4,
        max_seq_len=131072,
        sliding_window=128,
        reasoning_mode="hybrid",  # configurable reasoning effort (low/med/high)
    ),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": ModelConfig(
        name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        model_orig_dtype="bf16",
        ffn_weight_dtype="bf16",
        attn_weight_dtype="bf16",
        activation_dtype="bf16",
        kv_cache_dtype="bf16",
        n_layers=48,
        hidden_size=2048,
        vocab_size=151936,
        n_attention_heads=32,
        n_kv_heads=4,
        head_dim=128,
        intermediate_size=6144,
        moe_intermediate_size=768,
        n_experts=128,
        top_k=8,
        max_seq_len=262144,
        reasoning_mode="none",  # Coder Instruct is non-thinking
    ),
    "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit": ModelConfig(
        name="cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
        model_orig_dtype="bf16",
        ffn_weight_dtype="int4",
        attn_weight_dtype="int4",
        activation_dtype="bf16",
        kv_cache_dtype="bf16",
        n_layers=48,
        hidden_size=2048,
        vocab_size=151936,
        n_attention_heads=32,
        n_kv_heads=4,
        head_dim=128,
        intermediate_size=5472,
        moe_intermediate_size=768,
        n_experts=128,
        top_k=8,
        max_seq_len=262144,
        reasoning_mode="none",  # Coder Instruct is non-thinking
    ),
}
