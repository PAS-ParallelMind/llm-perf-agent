# Agent for LLM Inference Performance

An interactive chat agent for **LLM inference deployment guidance and
performance analysis**. The agent runs on any OpenAI-compatible
tool-calling endpoint (e.g. a vLLM server), and calls performance tools
on the user's behalf to answer questions like:

- "What GPU(s) do I need to deploy gpt-oss-20b at 4k context with concurrency 32?"
- "Is my workload compute- or memory-bound on H100? What's the TPOT?"
- "How does batch size 32 vs 128 change throughput for Qwen3-Coder-30B?"
- "Will this fit on a single H100 if I quantize the experts to MXFP4?"

The agent decides when to call a tool, interprets the result, and keeps
going across multiple turns of conversation.

## Project structure

```
agent/
  main.py            interactive REPL — entry point
  config.py          ChatConfig (agent model + session settings)
  loop.py            ChatAgent: multi-turn tool-calling loop
  engine.py          OpenAI-compatible client
  fake_engine.py     scripted engine for --dry-run
  prompts.py         system prompt + memory index
  memory.py          remember / recall tools, MEMORY.md
  types.py           SessionMeta dataclass
  workspace.py       session workspace (cwd for tools)
  tools/
    base.py          @tool registry + JSON-schema export
    fs.py            read / write / edit / glob / grep
    bash.py          sandboxed shell
    benchmarking/
      benchmark.py   ⏳ placeholder — probe a running inference endpoint
    modeling/
      memory.py      estimate_memory    — weights + KV cache VRAM breakdown
      latency.py     estimate_latency   — single forward-pass roofline
      serving.py     simulate_serving   — continuous-batching workload sim
      report.py      ReportBuilder      — shared text-report helpers
      configs/
        hw_specs.py    PRESET_GPUS      — 4090, A100, H100, H200, B200
        model_specs.py PRESET_MODELS    — gpt-oss-20b, Qwen3-Coder-30B (BF16 + AWQ-4bit)
runs/                per-session dirs (run.yaml + batch/session/...)
webui/               legacy web UI from the previous incarnation; useful
                     for inspecting session traces. Some pages depend on
                     artifacts that no longer exist and will be blank.
```

The `benchmarking/benchmark` tool is still a stub — it will eventually
probe a running OpenAI-compatible endpoint and return measured latency
and throughput. The three modeling tools under `modeling/` are real.

## Performance modeling tools

All three modeling tools take preset model and GPU names (see
`PRESET_MODELS` / `PRESET_GPUS`) and return a formatted text report.

### `estimate_memory`

Estimate GPU memory required to serve a model: weight bytes (respecting
per-component quantization: attention vs. FFN vs. embeddings) plus KV
cache bytes for the requested concurrency and context length. Sliding-
window attention layers are capped at the window size.

```
estimate_memory(model, concurrency, context_length) -> report
```

### `estimate_latency`

Roofline latency of a single transformer forward pass for a homogeneous
batch, broken down per operation (qkv_proj, attn_core, o_proj,
up_gate_proj, down_proj, lm_head) with a compute-vs-memory bottleneck
label per op.

```
estimate_latency(model, gpu, batch_size, input_tokens, kv_cache_len) -> report
```

For decode: `input_tokens=1`, `kv_cache_len = current context length`.
For prefill: `input_tokens = prompt length`, `kv_cache_len=0`.

### `simulate_serving`

Runs a fixed request pool through a vLLM-style continuous-batching
scheduler with a `max_num_batched_tokens` budget, accumulates per-phase
latencies, and reports TTFT / TPOT / throughput plus a per-op step
breakdown with bottleneck analysis.

```
simulate_serving(model, gpu, concurrency, input_len, output_len,
                 num_requests, max_num_batched_tokens) -> report
```

Each modeling module is also runnable as a standalone CLI for ad-hoc
checks, e.g.:

```bash
uv run python -m agent.tools.modeling.memory \
    --model openai/gpt-oss-20b --concurrency 32 --context-length 4096

uv run python -m agent.tools.modeling.serving \
    --model openai/gpt-oss-20b --gpu h100-sxm \
    --num-concurrency 16 --input-len 128 --output-len 64 --num-requests 64
```

## Quick start

```bash
# 1. Install deps
uv sync

# 2a. Dry-run the chat REPL with no model server (uses FakeEngine)
uv run python -m agent.main --dry-run

# 2b. Real run — point at any OpenAI-compatible endpoint
uv run python -m agent.main \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --base-url http://localhost:8000/v1
```

Slash commands inside the REPL:

```
/help      list commands
/tools     list registered tools
/reset     clear conversation history (keep the session dir)
/exit      quit
```

## Configuration via YAML

For repeatable sessions, drop a `chat.yaml`:

```yaml
agent:
  model:
    name:              openai/Qwen3-Coder-30B-A3B-Instruct
    base_url:          http://localhost:8001/v1
    api_key:           EMPTY
    temperature:       0.0
    max_output_tokens: 4096      # tokens generated per response
    max_model_len:     32768     # match the server's vLLM --max-model-len
    reasoning:         false
  max_steps:           20        # tool calls per user turn

session:
  dir:           runs/chat
  name:          null          # auto: chat-YYYYMMDD-HHMMSS

system_prompt: |
  You are an LLM inference performance engineer ...
```

```bash
uv run python -m agent.main --config chat.yaml
```

Note: the hardware / model *under analysis* is intentionally **not** in
config. Sessions typically evolve — the user asks "what GPU for
gpt-oss-20b?", agrees on hardware, then pivots to a different config.
The perf tools take their own parameters per call.

## Session outputs

Each session writes:

```
runs/<session-name>/
  run.yaml                       config snapshot
  batch/session/
    trace.json                   full message history (re-serialized each turn)
    tool_calls.jsonl             step-indexed tool log
    summary.json                 turns / steps / elapsed
    <any files the agent wrote>  bench outputs, scratch notes
```

Traces are crash-safe: re-serialized after every user turn.

## Memory

`memory/` is a persistent note store across sessions. `MEMORY.md` is
auto-loaded into the system prompt; the agent can call `remember` /
`recall` to write and read entries (preferred hardware, customer
constraints, known-good configs).

## Extending

### Add a new model or GPU preset

Edit `PRESET_MODELS` in [agent/tools/modeling/configs/model_specs.py](agent/tools/modeling/configs/model_specs.py)
or `PRESET_GPUS` in [agent/tools/modeling/configs/hw_specs.py](agent/tools/modeling/configs/hw_specs.py).
All three modeling tools will pick it up automatically.

### Wire up the benchmark tool

The `benchmark` placeholder lives at
[agent/tools/benchmarking/benchmark.py](agent/tools/benchmarking/benchmark.py).
Implementations should accept an endpoint + load shape and return
measured latency / throughput / TTFT / TPOT.

### Add a new tool

Drop a `@tool`-decorated function under `agent/tools/` and import the
module from `agent/tools/__init__.py` (or one of its subpackages). The
argument schema is derived from the function signature + type hints.
See [agent/tools/base.py](agent/tools/base.py) for the registry contract.

## See also

- [SPEC.md](SPEC.md) — chassis contracts (loop, tool registry, config)
- [AGENTS.md](AGENTS.md) — coding style + repo conventions
