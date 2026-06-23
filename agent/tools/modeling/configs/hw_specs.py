"""GPU preset specifications used by the modeling tools.

Cost model: ``cost_per_hour`` is **derived** from the GPU's MSRP and TDP
plus a small set of shared TCO assumptions (useful life, electricity
price, PUE). This is more defensible than rental quotes — anyone can
check MSRP + TDP against the datasheet, and the assumptions are explicit
and overridable. The result is an *owned-hardware* TCO, NOT a cloud
rental price (rental adds provider margin + utilisation risk on top).
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# TCO assumptions (override at module level if your deployment differs)
# ---------------------------------------------------------------------------

# Useful life over which capex is amortised.
USEFUL_LIFE_YEARS = 4.0
# Wholesale electricity (USD per kWh). US industrial average ~$0.08–0.10.
ELECTRICITY_USD_PER_KWH = 0.10
# Power Usage Effectiveness — data-centre overhead (cooling, distribution).
# Modern hyperscale ~1.1–1.2, typical enterprise ~1.3–1.5.
PUE = 1.3

_HOURS_PER_YEAR = 365 * 24


def derive_cost_per_hour(
    msrp_usd: float,
    tdp_watts: float,
    useful_life_years: float = USEFUL_LIFE_YEARS,
    electricity_usd_per_kwh: float = ELECTRICITY_USD_PER_KWH,
    pue: float = PUE,
) -> float:
    """Owned-hardware $/GPU-hour = amortised capex + electricity.

    Returns 0.0 if either MSRP or TDP is missing — callers check ``> 0``
    and omit cost lines when the input is incomplete."""
    if msrp_usd <= 0 or tdp_watts <= 0:
        return 0.0
    capex_per_h = msrp_usd / (useful_life_years * _HOURS_PER_YEAR)
    power_per_h = (tdp_watts / 1000.0) * pue * electricity_usd_per_kwh
    return capex_per_h + power_per_h


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    """Per-GPU compute throughput, memory, and pricing inputs.

    FLOP fields are dense, peak, no-sparsity throughput in operations/second.
    Memory bandwidth is HBM bytes/second; capacity is HBM bytes.
    ``msrp_usd`` is the GPU's approximate list price; ``tdp_watts`` is its
    thermal design power. The ``cost_per_hour`` property derives the
    owned-hardware TCO from these via :func:`derive_cost_per_hour`.
    """

    name: str
    bf16_flops: float
    mem_bandwidth: float
    mem_capacity: float
    fp8_flops: float = 0.0
    fp4_flops: float = 0.0
    mxfp4_flops: float = 0.0
    msrp_usd: float = 0.0
    tdp_watts: float = 0.0
    # Per-GPU NVLink bandwidth (bidirectional, GB/s). Drives the TP all-
    # reduce roofline in latency.py. 0.0 means "no fast interconnect" —
    # any TP > 1 simulation on such a part will see PCIe-bound comms,
    # which the roofline doesn't model, so callers should keep TP=1.
    nvlink_bw_gbps: float = 0.0

    @property
    def cost_per_hour(self) -> float:
        return derive_cost_per_hour(self.msrp_usd, self.tdp_watts)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# MSRPs are approximate list prices; older parts (A100) reflect current
# street/used pricing, which is well below original launch MSRP. TDPs are
# from the official datasheets.

PRESET_GPUS: dict[str, HardwareConfig] = {
    "rtx4090": HardwareConfig(
        name="rtx4090",
        bf16_flops=165.2e12,
        mem_bandwidth=1008e9,
        mem_capacity=24e9,
        fp8_flops=330.3e12,
        msrp_usd=1_600,
        tdp_watts=450,
    ),
    "rtx5090": HardwareConfig(
        name="rtx5090",
        bf16_flops=209e12,       # NVIDIA published spec, dense
        mem_bandwidth=1792e9,    # 1.79 TB/s GDDR7
        mem_capacity=32e9,       # 32 GB
        fp8_flops=419e12,        # ~2x BF16 (Blackwell ratio)
        fp4_flops=838e12,        # NVIDIA's "838 AI TOPS" headline, dense
        msrp_usd=1_999,          # Founders Edition launch MSRP
        tdp_watts=575,
    ),
    "h100-sxm": HardwareConfig(
        name="h100-sxm",
        bf16_flops=989e12,
        mem_bandwidth=3350e9,
        mem_capacity=80e9,
        fp8_flops=3958e12,
        msrp_usd=30_000,
        tdp_watts=700,
        nvlink_bw_gbps=450,      # NVLink 4: 18 links × 25 GB/s per direction
    ),
    "h100-pcie": HardwareConfig(
        name="h100-pcie",
        bf16_flops=756e12,
        mem_bandwidth=2000e9,
        mem_capacity=80e9,
        fp8_flops=1513e12,
        msrp_usd=25_000,
        tdp_watts=350,
        # NVL bridge is an optional accessory; default to "no fast link"
        # so a TP > 1 sim flags the missing comms model.
    ),
    "h200-nvl": HardwareConfig(
        name="h200-nvl",
        bf16_flops=835e12,
        mem_bandwidth=4800e9,
        mem_capacity=141e9,
        fp8_flops=1671e12,
        msrp_usd=32_000,
        tdp_watts=700,
        nvlink_bw_gbps=450,      # "NVL" = NVLink-enabled PCIe, NVLink 4 spec
    ),
    "a100-sxm": HardwareConfig(
        name="a100-sxm",
        bf16_flops=624e12,
        mem_bandwidth=2039e9,
        mem_capacity=80e9,
        fp8_flops=624e12,   # no native FP8; emulated at BF16 rate
        msrp_usd=10_000,    # well below 2020 launch MSRP — current street
        tdp_watts=400,
        nvlink_bw_gbps=300,      # NVLink 3: 12 links × 25 GB/s per direction
    ),
    "a100-pcie": HardwareConfig(
        name="a100-pcie",
        bf16_flops=624e12,
        mem_bandwidth=1935e9,
        mem_capacity=80e9,
        fp8_flops=624e12,
        msrp_usd=9_000,
        tdp_watts=300,
    ),
    "b200": HardwareConfig(
        name="b200",
        bf16_flops=2.25e15,
        mem_bandwidth=8e12,
        mem_capacity=180e9,
        fp8_flops=4.5e15,
        fp4_flops=9e15,
        msrp_usd=45_000,
        tdp_watts=1_000,
        nvlink_bw_gbps=900,      # NVLink 5: 18 links × 50 GB/s per direction
    ),
    # DGX Spark (GB10) — NVIDIA's compact "personal AI" workstation.
    # Two structural differences from the discrete-card presets above:
    #   * Memory is a unified LPDDR5x pool shared with the Grace CPU,
    #     NOT dedicated HBM — bandwidth is ~13x lower than H200 and any
    #     memory-bound LLM decode is dominated by it.
    #   * msrp_usd is the whole workstation (CPU + GPU + 128 GB RAM +
    #     chassis), not a card price — cost basis differs from above.
    # bf16/fp8 FLOPs are derived from Blackwell's standard 4:2:1 ratio
    # against NVIDIA's headline "1 PFLOPS sparse FP4" (≈ 500 TF dense).
    # TDP is the GPU portion only (full system draws ~240W at wall).
    "dgx-spark": HardwareConfig(
        name="dgx-spark",
        bf16_flops=125e12,       # estimated: FP4_dense / 4
        mem_bandwidth=273e9,     # LPDDR5x-8533, unified CPU+GPU pool
        mem_capacity=128e9,      # unified; CPU shares this pool
        fp8_flops=250e12,        # estimated: FP4_dense / 2
        fp4_flops=500e12,        # NVIDIA's "1 PFLOPS sparse" / 2
        msrp_usd=3_000,          # whole workstation
        tdp_watts=140,           # estimated GPU portion only
    ),
}
