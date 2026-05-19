"""GPU preset specifications used by the modeling tools."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareConfig:
    """Per-GPU compute throughput and memory specs.

    FLOP fields are dense, peak, no-sparsity throughput in operations/second.
    Memory bandwidth is HBM bytes/second; capacity is HBM bytes.
    """

    name: str
    bf16_flops: float
    mem_bandwidth: float
    mem_capacity: float
    fp8_flops: float = 0.0
    fp4_flops: float = 0.0
    mxfp4_flops: float = 0.0


PRESET_GPUS: dict[str, HardwareConfig] = {
    "4090": HardwareConfig(
        name="4090",
        bf16_flops=165.2e12,
        mem_bandwidth=1008e9,
        mem_capacity=24e9,
        fp8_flops=330.3e12,
    ),
    "h100-sxm": HardwareConfig(
        name="h100-sxm",
        bf16_flops=989e12,
        mem_bandwidth=3350e9,
        mem_capacity=80e9,
        fp8_flops=3958e12,
    ),
    "h100-pcie": HardwareConfig(
        name="h100-pcie",
        bf16_flops=756e12,
        mem_bandwidth=2000e9,
        mem_capacity=80e9,
        fp8_flops=1513e12,
    ),
    "h200-nvl": HardwareConfig(
        name="h200-nvl",
        bf16_flops=835e12,
        mem_bandwidth=4800e9,
        mem_capacity=141e9,
        fp8_flops=1671e12,
    ),
    "a100-sxm": HardwareConfig(
        name="a100-sxm",
        bf16_flops=624e12,
        mem_bandwidth=2039e9,
        mem_capacity=80e9,
        fp8_flops=624e12,  # no native FP8; emulated at BF16 rate
    ),
    "a100-pcie": HardwareConfig(
        name="a100-pcie",
        bf16_flops=624e12,
        mem_bandwidth=1935e9,
        mem_capacity=80e9,
        fp8_flops=624e12,
    ),
    "b200": HardwareConfig(
        name="b200",
        bf16_flops=2.25e15,
        mem_bandwidth=8e12,
        mem_capacity=180e9,
        fp8_flops=4.5e15,
        fp4_flops=9e15,
    ),
}
