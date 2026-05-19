"""Benchmark tool — placeholder.

Will eventually probe a running OpenAI-compatible endpoint (or wrap a
vLLM benchmark script) and return latency / throughput / TTFT / TPOT.
The argument schema below is intentionally loose; tighten it when the
implementation lands.
"""
from __future__ import annotations

from ..base import tool


@tool(
    "Run a latency / throughput benchmark against an LLM inference "
    "endpoint and return summary metrics. NOTE: this tool is a "
    "placeholder; the actual probe is not implemented yet.",
    spec="JSON object describing what to benchmark "
         "(endpoint, model, concurrency, prompt set, ...). "
         "Exact schema TBD.",
)
def benchmark(spec: str = "") -> str:
    return (
        "benchmark(): not implemented yet. "
        "When wired up, this will accept an endpoint + model + load shape "
        "and return latency / throughput / TTFT / TPOT statistics. "
        "For now, fall back to reasoning analytically or ask the user for "
        "measured numbers."
    )
