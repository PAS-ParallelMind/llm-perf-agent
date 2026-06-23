"""Per-forward-pass latency model — two modes (theoretical / baseline).

The previous version of this module computed per-op latencies analytically
from FLOPs/bytes and the GPU's theoretical peaks, scaled by an optional
``efficiency_factor`` scalar. That approach over-predicted throughput by
5-13x on B200 mxfp4 (validated empirically — see ``workspace/case1_v2``).
This module replaces it with a microbench-driven predictor whose
per-shape efficiency is read from ``measurements/microbenchmarks/<gpu>/``.

Two modes (``mode=`` arg, also surfaced as ``latency_source`` in
serving.py):

* ``"theoretical"`` — analytic FLOPs/bytes against the GPU's
  theoretical peaks (efficiency = 1.0 for every op). The optimistic
  ceiling — "what could the hardware reach if every kernel hit peak?".
  Over-predicts throughput by ~5-10x on real hardware.
* ``"baseline"`` (default) — same roofline math, but each op's wall
  time is divided by a per-shape efficiency factor interpolated from
  the llm-gpu-bench microbench grid. This is the realistic projection
  — tracks measured TPOT within ~15% across N=1..256 at in=6144/out=1024
  on B200; ~13% on H200.

Continuous batching: a step's running queue is split into decoders
(``tokens_this_step==1`` and prefill complete) and prefill chunks
(everyone else). The vLLM FlashInfer backend launches BatchDecode +
BatchPrefill kernels back-to-back on one stream, so per-step time is
the sum of the two — validated to ~1.8% mean in llm-gpu-bench.

SWA prefill (``Sq > sliding_window``) is a documented coverage gap in
the prefill grid: the sweep enforces ``Sk >= Sq``, so SWA prefill cells
fall back to the analytic roofline. Affects TTFT on SWA models like
gpt-oss-20b; TPOT (decode) is unaffected.

Tensor parallelism (``tp``): each rank owns 1/TP of the attention heads
(Q and KV) and 1/TP of the FFN/MoE intermediate dim. Each layer's two
row-parallel matmuls (O-proj, FFN/MoE down-proj) are followed by a
ring all-reduce whose wall time is modeled by the unidirectional NVLink
bandwidth on the GPU spec (``StepBreakdown.comm_s``). TP > 1 requires
``n_attention_heads`` and ``n_kv_heads`` both divisible by TP and a
non-zero ``HardwareConfig.nvlink_bw_gbps``.

Data parallelism (``dp``) is not modeled here — replicas are independent
and share no tensors per forward pass, so DP is a serving-level concern
(see ``serving.run_simulation``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .predict import Predictor
from .configs.model_specs import ModelConfig, get_quantization_bytes
from .configs.hw_specs import PRESET_GPUS


MODES = ("baseline", "theoretical")
# Hardware profile JSONs ship in-tree alongside hw_specs.py / model_specs.py
# — they characterize the GPU's per-kernel throughput, same flavor of static
# config as the rest of configs/.
HW_PROFILES_ROOT = Path(__file__).parent / "configs" / "hw_profiles"


# ---------------------------------------------------------------------------
# Parallelism configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParallelConfig:
    """Per-replica parallelism strategy for the serving deployment.

    * ``tp`` — tensor-parallel degree within one replica. Each rank owns
      1/TP of the attention heads (Q and KV) and 1/TP of the FFN/MoE
      intermediate dim; row-parallel matmuls are followed by an
      all-reduce (see ``_transformer_layer_s``).
    * ``dp`` — data-parallel replication. Independent replicas, no
      cross-replica tensors during a forward pass — only the
      serving-level workload (rate / n_requests) is sharded.

    Total GPU count for the deployment is ``tp * dp``.
    """
    tp: int = 1
    dp: int = 1

    @property
    def n_gpus(self) -> int:
        return self.tp * self.dp

    def validate(self, model: "ModelConfig") -> None:
        """Reject parallel degrees that don't divide the partitionable
        dimensions evenly. ``tp > n_kv_heads`` is allowed: each KV head is
        replicated across ``tp / n_kv_heads`` ranks so attention still
        partitions cleanly by Q heads."""
        if self.tp < 1 or self.dp < 1:
            raise ValueError(
                f"tp and dp must be >= 1, got tp={self.tp} dp={self.dp}")
        if model.n_attention_heads % self.tp != 0:
            raise ValueError(
                f"tp={self.tp} does not divide "
                f"n_attention_heads={model.n_attention_heads}")
        # tp > n_kv_heads triggers KV-head replication, modeled by clamping
        # per-rank n_kv to 1 (see _transformer_layer_s). Cluster-wide KV
        # cache scales by tp/n_kv_heads (handled in serving._kv_bytes_per_token_summed).


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

    ``nvlink_bw_gbps`` is the GPU's unidirectional NVLink throughput; it
    feeds the TP all-reduce roofline in ``_allreduce_ms``. 0.0 means the
    part has no fast inter-GPU link, so TP > 1 sims will be PCIe-bound —
    callers should keep TP=1 on such parts.
    """
    gpu_name: str
    nvlink_bw_gbps: float
    gemm_bf16: Predictor | None
    attn_bf16: Predictor | None
    moe_bf16: Predictor | None
    moe_mxfp4: Predictor | None
    allreduce: Predictor | None

    @classmethod
    def from_gpu(cls, gpu_name: str, root: Path | None = None) -> "PredictorBundle":
        base = (root or HW_PROFILES_ROOT) / gpu_name

        def _load(stem: str) -> Predictor | None:
            p = base / f"{stem}.json"
            return Predictor.from_json(p) if p.exists() else None

        gpu = PRESET_GPUS.get(gpu_name)
        nvlink_bw = gpu.nvlink_bw_gbps if gpu is not None else 0.0

        return cls(
            gpu_name=gpu_name,
            nvlink_bw_gbps=nvlink_bw,
            gemm_bf16=_load("gemm_bf16"),
            attn_bf16=_load("attn_bf16"),
            moe_bf16=_load("moe_bf16"),
            moe_mxfp4=_load("moe_mxfp4"),
            allreduce=_load("allreduce"),
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
    """Per-forward-pass latency components in seconds.

    Every field is the total contribution across a whole forward pass:
    the per-layer ops (qkv / attn_* / o_proj / ffn_or_moe / comm) are
    summed across all ``n_layers``; ``lm_head_s`` is the single lm-head
    matmul run once per pass on the rows being sampled.

    ``comm_s`` is the TP all-reduce wall time across both collectives per
    layer summed across all layers; zero when TP=1.
    """
    qkv_s: float = 0.0
    attn_decode_s: float = 0.0
    attn_prefill_s: float = 0.0
    o_proj_s: float = 0.0
    ffn_or_moe_s: float = 0.0
    comm_s: float = 0.0
    lm_head_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (self.qkv_s + self.attn_decode_s + self.attn_prefill_s
                + self.o_proj_s + self.ffn_or_moe_s + self.comm_s
                + self.lm_head_s)

    def iter_ops(self) -> Iterator[tuple[str, float]]:
        for name in ("qkv", "attn_decode", "attn_prefill", "o_proj",
                     "ffn_or_moe", "comm", "lm_head"):
            yield name, getattr(self, f"{name}_s")

    def accumulate(self, other: "StepBreakdown") -> None:
        self.qkv_s += other.qkv_s
        self.attn_decode_s += other.attn_decode_s
        self.attn_prefill_s += other.attn_prefill_s
        self.o_proj_s += other.o_proj_s
        self.ffn_or_moe_s += other.ffn_or_moe_s
        self.comm_s += other.comm_s
        self.lm_head_s += other.lm_head_s


# ---------------------------------------------------------------------------
# Per-op latencies — mode='theoretical' returns the analytic floor,
# mode='baseline' divides it by the grid's per-shape efficiency.
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


def _allreduce_ms(tp: int, M: int, hidden: int, act_bytes: float,
                  nvlink_bw_gbps: float,
                  predictor: Predictor | None = None) -> float:
    """All-reduce wall time for one collective, in milliseconds.

    When ``predictor`` carries measured (world_size, bytes) → latency
    points, those are used directly via the predictor's bilinear log-log
    interpolation. Otherwise falls back to a pure-bandwidth ring formula
    against the GPU's peak unidirectional NVLink throughput.

    Returns 0 for TP=1 (no comm) or when the fallback NVLink bandwidth
    is also 0 — the latter means the part has no fast inter-GPU link and
    callers should not be running TP > 1 on it.
    """
    if tp <= 1:
        return 0.0
    bytes_payload = int(M * hidden * act_bytes)
    if predictor is not None and predictor.ar_lat_ms:
        lat = predictor.allreduce_latency_ms(tp, bytes_payload)
        if lat == lat:               # NaN-safe
            return lat
    if nvlink_bw_gbps <= 0:
        return 0.0
    bw_bps = nvlink_bw_gbps * 1e9
    return 2 * (tp - 1) / tp * bytes_payload / bw_bps * 1000.0


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
    tp: int = 1,
) -> StepBreakdown:
    """One transformer layer's latency for one forward pass. Splits the
    batch into decode (Sq=1) and prefill (Sq>1) buckets per FlashInfer's
    additive composition.

    Under TP=N each rank owns 1/N of the attention heads (Q and KV) and
    1/N of the FFN/MoE intermediate dim. The two row-parallel matmuls
    (O-proj and FFN/MoE down-proj) are followed by an all-reduce; these
    sum into ``StepBreakdown.comm_s``.
    """
    hidden = model.hidden_size
    n_qo = model.n_attention_heads // tp
    # When tp > n_kv_heads, KV heads are replicated across ranks so each
    # rank still owns at least one KV head (K/V projection weights are
    # replicated; per-rank compute and per-rank KV cache stay at "1 KV head
    # worth"). Cluster-wide KV cache scales by tp/n_kv_heads — accounted
    # for in serving._kv_bytes_per_token_summed.
    n_kv = max(1, model.n_kv_heads // tp)
    head_dim = model.head_dim
    q_proj_dim = head_dim * n_qo
    qkv_out = head_dim * (n_qo + 2 * n_kv)
    intermediate = model.intermediate_size // tp
    moe_intermediate = model.moe_intermediate_size // tp
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
                            hidden, moe_intermediate,
                            weight_dtype=model.ffn_weight_dtype,
                            roofline_only=roofline_only)
    else:
        t_up = _gemm_ms(bundle, M, hidden, 2 * intermediate,
                        model.ffn_weight_dtype, roofline_only=roofline_only)
        t_down = _gemm_ms(bundle, M, intermediate, hidden,
                          model.ffn_weight_dtype, roofline_only=roofline_only)
        t_ffn_ms = t_up + t_down

    # Two row-parallel collectives per layer: one after attention O-proj,
    # one after FFN/MoE down-proj. Zero when TP=1. Uses the measured
    # all-reduce predictor when present; falls back to the NVLink ring
    # roofline otherwise.
    act_bytes = get_quantization_bytes(model.activation_dtype)
    t_comm_ms = 2 * _allreduce_ms(tp, M, hidden, act_bytes,
                                   bundle.nvlink_bw_gbps,
                                   predictor=bundle.allreduce)

    return StepBreakdown(
        qkv_s=t_qkv_ms / 1000.0,
        attn_decode_s=t_dec_ms / 1000.0,
        attn_prefill_s=t_pre_ms / 1000.0,
        o_proj_s=t_o_ms / 1000.0,
        ffn_or_moe_s=t_ffn_ms / 1000.0,
        comm_s=t_comm_ms / 1000.0,
    )


def forward_pass_latency(
    running_queue: list[Request],
    gpu_name: str,
    model: ModelConfig,
    *,
    mode: str = "baseline",
    parallel: ParallelConfig = ParallelConfig(),
) -> StepBreakdown:
    """Compute one forward pass's per-op latency.

    Extracts the decode / prefill split from the running queue's
    ``tokens_this_step`` and ``kv_tokens`` fields (set by the scheduler),
    then sums per-layer contributions. ``mode`` controls whether
    efficiency factors come from the microbench grid (``"baseline"``)
    or are forced to 1.0 (``"theoretical"``). Only ``parallel.tp``
    shapes the per-pass math — DP is a serving-level concern, so
    ``parallel.dp`` is unused here.
    """
    if mode not in MODES:
        raise ValueError(f"mode={mode!r} must be one of {MODES}")
    parallel.validate(model)
    bundle = get_bundle(gpu_name)
    roofline_only = (mode == "theoretical")

    decode_kvs: list[int] = []
    prefill_chunks: list[tuple[int, int]] = []
    n_lm_head_rows = 0
    for r in running_queue:
        if r.tokens_this_step == 0:
            continue
        if r.tokens_this_step == 1 and r.kv_tokens >= r.prompt_tokens:
            decode_kvs.append(r.kv_tokens)
            n_lm_head_rows += 1                 # decoder produces 1 sampled logit
        else:
            sq = r.tokens_this_step
            sk = r.kv_tokens + sq
            prefill_chunks.append((sq, sk))
            if sk == r.prompt_tokens:
                n_lm_head_rows += 1             # final prefill chunk → first sampled logit

    total = StepBreakdown()
    for layer_idx in range(model.n_layers):
        layer = _transformer_layer_s(
            bundle, model, layer_idx,
            decode_tokens=decode_kvs, prefill_chunks=prefill_chunks,
            roofline_only=roofline_only, tp=parallel.tp,
        )
        total.accumulate(layer)

    # lm_head runs ONCE per forward pass on rows being sampled (decoders +
    # final prefill chunk owners). Column-parallel under TP: each rank owns
    # vocab_size/tp output columns. No comm modeled — vLLM-style top-k +
    # reduce avoids a full vocab all-gather, and the residual reduce cost
    # is dwarfed by the other comm terms.
    if n_lm_head_rows > 0:
        vocab_per_rank = model.vocab_size // parallel.tp
        total.lm_head_s = _gemm_ms(
            bundle, n_lm_head_rows, model.hidden_size, vocab_per_rank,
            model.model_orig_dtype, roofline_only=roofline_only,
        ) / 1000.0
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
    parallel: ParallelConfig = ParallelConfig(),
) -> tuple[float, dict[str, float]]:
    """Direct entry: pass decode/prefill batch composition explicitly.

    Returns (total_ms, breakdown_dict_in_ms_per_op). Used by the standalone
    sims in ``_sim_v2_*``. Production callers (serving.py) go through
    ``forward_pass_latency`` instead.
    """
    parallel.validate(model)
    total = StepBreakdown()
    for layer_idx in range(model.n_layers):
        layer = _transformer_layer_s(
            bundle, model, layer_idx,
            decode_tokens=list(decode_tokens), prefill_chunks=list(prefill_chunks),
            roofline_only=roofline_only, tp=parallel.tp,
        )
        total.accumulate(layer)
    # Approximate: assume every decoder + every prefill chunk samples a logit.
    # (Production calls go through forward_pass_latency which knows which
    # prefill chunks are the final one and skips lm_head for mid-prefill.)
    n_lm_head_rows = len(decode_tokens) + len(prefill_chunks)
    if n_lm_head_rows > 0:
        vocab_per_rank = model.vocab_size // parallel.tp
        total.lm_head_s = _gemm_ms(
            bundle, n_lm_head_rows, model.hidden_size, vocab_per_rank,
            model.model_orig_dtype, roofline_only=roofline_only,
        ) / 1000.0
    return total.total_s * 1000.0, {
        "qkv": total.qkv_s * 1000.0,
        "attn_decode": total.attn_decode_s * 1000.0,
        "attn_prefill": total.attn_prefill_s * 1000.0,
        "o_proj": total.o_proj_s * 1000.0,
        "ffn_or_moe": total.ffn_or_moe_s * 1000.0,
        "comm": total.comm_s * 1000.0,
        "lm_head": total.lm_head_s * 1000.0,
    }
