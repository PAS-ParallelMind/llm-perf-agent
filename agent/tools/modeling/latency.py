"""Per-forward-pass latency model — two modes (roofline / predictor).

The previous version of this module computed per-op latencies analytically
from FLOPs/bytes and the GPU's theoretical peaks, scaled by an optional
``efficiency_factor`` scalar. That approach over-predicted throughput by
5-13x on B200 mxfp4 (validated empirically — see ``workspace/case1_v2``).
This module replaces it with a microbench-driven predictor whose
per-shape efficiency is read from ``measurements/microbenchmarks/<gpu>/``.

Two modes:

* ``"roofline"`` — analytic FLOPs/bytes against the GPU's theoretical
  peaks (efficiency = 1.0 for every op). Useful as a baseline /
  ablation; for B200 this over-predicts throughput by ~7x.
* ``"predictor"`` (default) — same roofline math, but each op's wall
  time is divided by a per-shape efficiency factor interpolated from
  the llm-gpu-bench microbench grid. On B200 this tracks measured TPOT
  within ~15% across N=1..256 at in=6144/out=1024.

Continuous batching: a step's running queue is split into decoders
(``tokens_this_step==1`` and prefill complete) and prefill chunks
(everyone else). The vLLM FlashInfer backend launches BatchDecode +
BatchPrefill kernels back-to-back on one stream, so per-step time is
the sum of the two — validated to ~1.8% mean in llm-gpu-bench.

SWA prefill (``Sq > sliding_window``) is a documented coverage gap in
the prefill grid: the sweep enforces ``Sk >= Sq``, so SWA prefill cells
fall back to the analytic roofline. Affects TTFT on SWA models like
gpt-oss-20b; TPOT (decode) is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .predict import Predictor
from .configs.model_specs import ModelConfig


MODES = ("roofline", "predictor")
# Hardware profile JSONs ship in-tree alongside hw_specs.py / model_specs.py
# — they characterize the GPU's per-kernel throughput, same flavor of static
# config as the rest of configs/.
HW_PROFILES_ROOT = Path(__file__).parent / "configs" / "hw_profiles"


# ---------------------------------------------------------------------------
# Per-request scheduler state (consumed by serving.run_simulation)
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """One in-flight request tracked by the serving simulator.

    The four timestamps below are set by the scheduler and define every
    derived per-request latency:

    * ``ttft  = first_token_s - arrival_s``
    * ``tpot  = (finish_s - first_token_s) / max(gen_tokens - 1, 1)``
    * ``e2e   = finish_s - arrival_s``
    * ``wait  = start_s - arrival_s``
    """

    id: Any
    arrival_s: float
    prompt_tokens: int
    gen_tokens: int
    kv_tokens: int = 0           # KV cached before the current step
    tokens_this_step: int = 0    # how many tokens this request runs THIS step
    start_s: float = 0.0         # first scheduled into running queue
    first_token_s: float = 0.0   # end of step where prefill completed
    finish_s: float = 0.0        # end of step where last decode ran


def total_tokens_in_batch(requests: list[Request]) -> int:
    return sum(r.tokens_this_step for r in requests)


# ---------------------------------------------------------------------------
# Per-GPU predictor bundle
# ---------------------------------------------------------------------------

@dataclass
class PredictorBundle:
    """Per-GPU predictor set covering the kernel surface we model.
    Any field may be ``None`` if its JSON is absent — callers should
    check before using a specific op. GEMM is bf16/fp16 only.
    """
    gpu_name: str
    gemm_bf16: Predictor | None
    attn_bf16: Predictor | None
    moe_bf16: Predictor | None
    moe_mxfp4: Predictor | None

    @classmethod
    def from_gpu(cls, gpu_name: str, root: Path | None = None) -> "PredictorBundle":
        base = (root or HW_PROFILES_ROOT) / gpu_name

        def _load(stem: str) -> Predictor | None:
            p = base / f"{stem}.json"
            return Predictor.from_json(p) if p.exists() else None

        return cls(
            gpu_name=gpu_name,
            gemm_bf16=_load("gemm_bf16"),
            attn_bf16=_load("attn_bf16"),
            moe_bf16=_load("moe_bf16"),
            moe_mxfp4=_load("moe_mxfp4"),
        )

    def gemm(self) -> Predictor:
        if self.gemm_bf16 is None:
            raise RuntimeError(f"no gemm_bf16 predictor for {self.gpu_name}")
        return self.gemm_bf16

    def moe(self, weight_dtype: str) -> Predictor:
        p = self.moe_mxfp4 if weight_dtype == "mxfp4" else self.moe_bf16
        if p is None:
            raise RuntimeError(
                f"no moe_{weight_dtype} predictor for {self.gpu_name}")
        return p


_BUNDLES: dict[str, PredictorBundle] = {}


def get_bundle(gpu_name: str) -> PredictorBundle:
    """Cached lookup of the per-GPU predictor set."""
    if gpu_name not in _BUNDLES:
        _BUNDLES[gpu_name] = PredictorBundle.from_gpu(gpu_name)
    return _BUNDLES[gpu_name]


# ---------------------------------------------------------------------------
# Per-step latency breakdown (consumed by serving.summarize_run)
# ---------------------------------------------------------------------------

@dataclass
class StepBreakdown:
    """Per-step latency components in seconds. Replaces the old
    ``OpBreakdown``: no compute/memory split (the predictor returns a
    single number per op), so the per-op report becomes just
    ``Op | Time(ms) | %Step``."""
    qkv_s: float = 0.0
    attn_decode_s: float = 0.0
    attn_prefill_s: float = 0.0
    o_proj_s: float = 0.0
    ffn_or_moe_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (self.qkv_s + self.attn_decode_s + self.attn_prefill_s
                + self.o_proj_s + self.ffn_or_moe_s)

    def iter_ops(self) -> Iterator[tuple[str, float]]:
        for name in ("qkv", "attn_decode", "attn_prefill", "o_proj", "ffn_or_moe"):
            yield name, getattr(self, f"{name}_s")

    def accumulate(self, other: "StepBreakdown") -> None:
        self.qkv_s += other.qkv_s
        self.attn_decode_s += other.attn_decode_s
        self.attn_prefill_s += other.attn_prefill_s
        self.o_proj_s += other.o_proj_s
        self.ffn_or_moe_s += other.ffn_or_moe_s


# ---------------------------------------------------------------------------
# Per-op latencies — mode='roofline' returns the analytic floor,
# mode='predictor' divides it by the grid's per-shape efficiency.
# ---------------------------------------------------------------------------

def _gemm_ms(bundle: PredictorBundle, M: int, K: int, N: int,
             weight_dtype: str, *, roofline_only: bool) -> float:
    """Dense GEMM latency. Only bf16/fp16 weights are supported."""
    if weight_dtype not in ("bf16", "fp16"):
        raise NotImplementedError(f"gemm only supports bf16/fp16; got {weight_dtype}")
    pred = bundle.gemm()
    return pred.roofline_ms(M, K, N, "bf16") if roofline_only else pred.latency_ms(M, K, N, "bf16")


def _attn_decode_ms(bundle: PredictorBundle, R: int, kv_tokens_total: int,
                    n_qo: int, n_kv: int, head_dim: int,
                    *, roofline_only: bool) -> float:
    if bundle.attn_bf16 is None:
        raise RuntimeError(f"no attn_bf16 predictor for {bundle.gpu_name}")
    Sk = kv_tokens_total // max(R, 1)
    if roofline_only:
        return bundle.attn_bf16.attn_roofline_ms(R=R, Sq=1, Sk=Sk,
                                                  H=n_qo, H_kv=n_kv, D=head_dim)
    return bundle.attn_bf16.attn_latency_ms(R=R, Sq=1, Sk=Sk,
                                             H=n_qo, H_kv=n_kv, D=head_dim)


def _attn_prefill_ms(bundle: PredictorBundle, R: int, Sq: int, Sk: int,
                     n_qo: int, n_kv: int, head_dim: int,
                     sliding_window: int | None = None,
                     *, roofline_only: bool) -> float:
    if bundle.attn_bf16 is None:
        raise RuntimeError(f"no attn_bf16 predictor for {bundle.gpu_name}")
    pred = bundle.attn_bf16
    # SWA prefill (Sq > window) is a grid coverage gap: always falls back
    # to the roofline regardless of mode.
    if sliding_window is not None and Sq > sliding_window:
        return pred.attn_roofline_ms(R=R, Sq=Sq, Sk=sliding_window,
                                      H=n_qo, H_kv=n_kv, D=head_dim)
    if roofline_only:
        return pred.attn_roofline_ms(R=R, Sq=Sq, Sk=Sk,
                                      H=n_qo, H_kv=n_kv, D=head_dim)
    return pred.attn_latency_ms(R=R, Sq=Sq, Sk=Sk,
                                 H=n_qo, H_kv=n_kv, D=head_dim)


def _moe_ms(bundle: PredictorBundle, M: int, n_experts: int, top_k: int,
            hidden: int, intermediate: int, weight_dtype: str,
            *, roofline_only: bool) -> float:
    pred = bundle.moe(weight_dtype)
    if roofline_only:
        return pred.moe_roofline_ms(M, n_experts, top_k, hidden, intermediate)
    return pred.moe_latency_ms(M, n_experts, top_k, hidden, intermediate)


# ---------------------------------------------------------------------------
# Layer + forward-pass composition
# ---------------------------------------------------------------------------

def _transformer_layer_s(
    bundle: PredictorBundle,
    model: ModelConfig,
    layer_idx: int,
    *,
    decode_tokens: list[int],
    prefill_chunks: list[tuple[int, int]],
    roofline_only: bool,
) -> StepBreakdown:
    """One transformer layer's latency for one forward pass. Splits the
    batch into decode (Sq=1) and prefill (Sq>1) buckets per FlashInfer's
    additive composition."""
    hidden = model.hidden_size
    n_qo = model.n_attention_heads
    n_kv = model.n_kv_heads
    head_dim = model.head_dim
    q_proj_dim = head_dim * n_qo
    qkv_out = head_dim * (n_qo + 2 * n_kv)
    swa = model.sliding_window if model.layer_uses_sliding_window(layer_idx) else None

    n_decode = len(decode_tokens)
    n_prefill = sum(sq for sq, _ in prefill_chunks)
    M = n_decode + n_prefill
    if M == 0:
        return StepBreakdown()

    t_qkv_ms = _gemm_ms(bundle, M, hidden, qkv_out, model.attn_weight_dtype,
                        roofline_only=roofline_only)
    t_o_ms = _gemm_ms(bundle, M, q_proj_dim, hidden, model.attn_weight_dtype,
                      roofline_only=roofline_only)

    t_dec_ms = 0.0
    if n_decode > 0:
        kv_capped = [min(k, swa) if swa is not None else k for k in decode_tokens]
        kv_total = sum(kv_capped)
        t_dec_ms = _attn_decode_ms(bundle, n_decode, kv_total, n_qo, n_kv, head_dim,
                                    roofline_only=roofline_only)

    t_pre_ms = 0.0
    for sq, sk in prefill_chunks:
        sk_capped = min(sk, swa) if swa is not None else sk
        t_pre_ms += _attn_prefill_ms(bundle, 1, sq, sk_capped, n_qo, n_kv, head_dim,
                                      sliding_window=swa,
                                      roofline_only=roofline_only)

    if model.n_experts > 1:
        t_ffn_ms = _moe_ms(bundle, M, model.n_experts, model.top_k,
                            hidden, model.moe_intermediate_size,
                            weight_dtype=model.ffn_weight_dtype,
                            roofline_only=roofline_only)
    else:
        t_up = _gemm_ms(bundle, M, hidden, 2 * model.intermediate_size,
                        model.ffn_weight_dtype, roofline_only=roofline_only)
        t_down = _gemm_ms(bundle, M, model.intermediate_size, hidden,
                          model.ffn_weight_dtype, roofline_only=roofline_only)
        t_ffn_ms = t_up + t_down

    return StepBreakdown(
        qkv_s=t_qkv_ms / 1000.0,
        attn_decode_s=t_dec_ms / 1000.0,
        attn_prefill_s=t_pre_ms / 1000.0,
        o_proj_s=t_o_ms / 1000.0,
        ffn_or_moe_s=t_ffn_ms / 1000.0,
    )


def forward_pass_latency(
    running_queue: list[Request],
    gpu_name: str,
    model: ModelConfig,
    *,
    mode: str = "predictor",
) -> StepBreakdown:
    """Compute one forward pass's per-op latency.

    Extracts the decode / prefill split from the running queue's
    ``tokens_this_step`` and ``kv_tokens`` fields (set by the scheduler),
    then sums per-layer contributions. ``mode`` controls whether
    efficiency factors come from the microbench grid (``"predictor"``)
    or are forced to 1.0 (``"roofline"``).
    """
    if mode not in MODES:
        raise ValueError(f"mode={mode!r} must be one of {MODES}")
    bundle = get_bundle(gpu_name)
    roofline_only = (mode == "roofline")

    decode_kvs: list[int] = []
    prefill_chunks: list[tuple[int, int]] = []
    for r in running_queue:
        if r.tokens_this_step == 0:
            continue
        if r.tokens_this_step == 1 and r.kv_tokens >= r.prompt_tokens:
            decode_kvs.append(r.kv_tokens)
        else:
            sq = r.tokens_this_step
            sk = r.kv_tokens + sq
            prefill_chunks.append((sq, sk))

    total = StepBreakdown()
    for layer_idx in range(model.n_layers):
        layer = _transformer_layer_s(
            bundle, model, layer_idx,
            decode_tokens=decode_kvs, prefill_chunks=prefill_chunks,
            roofline_only=roofline_only,
        )
        total.accumulate(layer)
    return total


# ---------------------------------------------------------------------------
# Lower-level entry points (kept for tests / _sim_v2_* shims)
# ---------------------------------------------------------------------------

def forward_pass_ms(
    bundle: PredictorBundle,
    model: ModelConfig,
    *,
    decode_tokens: list[int],
    prefill_chunks: list[tuple[int, int]] = (),
    roofline_only: bool = False,
) -> tuple[float, dict[str, float]]:
    """Direct entry: pass decode/prefill batch composition explicitly.

    Returns (total_ms, breakdown_dict_in_ms_per_op). Used by the standalone
    sims in ``_sim_v2_*``. Production callers (serving.py) go through
    ``forward_pass_latency`` instead.
    """
    total = StepBreakdown()
    for layer_idx in range(model.n_layers):
        layer = _transformer_layer_s(
            bundle, model, layer_idx,
            decode_tokens=list(decode_tokens), prefill_chunks=list(prefill_chunks),
            roofline_only=roofline_only,
        )
        total.accumulate(layer)
    return total.total_s * 1000.0, {
        "qkv": total.qkv_s * 1000.0,
        "attn_decode": total.attn_decode_s * 1000.0,
        "attn_prefill": total.attn_prefill_s * 1000.0,
        "o_proj": total.o_proj_s * 1000.0,
        "ffn_or_moe": total.ffn_or_moe_s * 1000.0,
    }
