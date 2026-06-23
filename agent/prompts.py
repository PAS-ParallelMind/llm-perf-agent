"""System prompt assembly.

Task-specific guidance lives in ``chat.yaml`` (``system_prompt`` field).
This module only handles (a) a minimal default fallback aimed at the
LLM-inference-perf use case, and (b) appending the persistent memory
index to whatever task prompt was given.
"""
from __future__ import annotations

_MEMORY_BLOCK = """
## Memory
Persistent notes live under ./memory. Use `remember` to save durable
facts (preferred hardware, customer constraints, known-good configs)
and `recall` to re-read a file by name.

### MEMORY.md
{memory_index}
"""

_DEFAULT_TASK_PROMPT = (
    "You are an LLM inference performance engineer. Help the user with "
    "deployment hardware guidance and performance analysis for serving "
    "large language models.\n\n"
    "When the user asks a question, decide whether you can answer "
    "directly or whether a tool call would yield a more precise answer. "
    "Available perf tools: `estimate_memory` (weights + KV cache VRAM "
    "fit), `simulate_serving` (continuous-batching TTFT/TPOT/throughput — "
    "modeled), and `benchmark_serving` (the MEASURED counterpart: drives "
    "a real load through a running vLLM/OpenAI-compatible server via "
    "`vllm bench serve` and reports measured TTFT/TPOT/throughput). "
    "`benchmark_serving` mirrors `simulate_serving`'s parameters, so use "
    "it to ground-truth a modeled estimate; it records results to the "
    "measurement store, which `lookup_measurements` reads back to "
    "calibrate future theory. After a tool returns, interpret the result "
    "for the user — don't just dump raw numbers."
)


def build_system_prompt(
    memory_index: str,
    task_prompt: str | None = None,
) -> str:
    """Compose the final system prompt from a task prompt + memory index."""
    task = (task_prompt or _DEFAULT_TASK_PROMPT).rstrip()
    memory = _MEMORY_BLOCK.format(memory_index=memory_index.strip() or "(empty)")
    return f"{task}\n{memory}"
