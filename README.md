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
      benchmark.py     benchmark_serving — MEASURED TTFT/TPOT/throughput
                       (wraps `vllm bench serve` against a live endpoint)
      measurements.py  record_measurement / lookup_measurements store
    modeling/
      memory.py      estimate_memory    — weights + KV cache VRAM breakdown
      latency.py     (plumbing)          — per-forward-pass latency model used by serving.py
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

The modeling tools under `modeling/` produce simulated numbers — either
microbench-calibrated (`baseline` mode, the realistic projection) or
roofline (`theoretical` mode, the optimistic ceiling). The
`benchmarking/benchmark_serving` tool is the **measured** counterpart:
it drives a synthetic workload through a *running* OpenAI-compatible
server with `vllm bench serve` and reports real TTFT / TPOT / throughput,
then records the result so estimates can be calibrated against it.

## Performance modeling tools

Both modeling tools take preset model and GPU names (see
`PRESET_MODELS` / `PRESET_GPUS`) and return a formatted text report.

### `estimate_memory`

Estimate GPU memory required to serve a model: weight bytes (respecting
per-component quantization: attention vs. FFN vs. embeddings) plus KV
cache bytes for the requested concurrency and context length. Sliding-
window attention layers are capped at the window size.

```
estimate_memory(model, concurrency, context_length) -> report
```

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

## Measured benchmarking tool

### `benchmark_serving`

The measured counterpart to `simulate_serving`. Drives a synthetic
`random` workload through a **running** OpenAI-compatible server (e.g.
vLLM) with `vllm bench serve` and reports real TTFT / TPOT / ITL / E2EL
(mean / median / p99) plus request and token throughput. The parameters
mirror `simulate_serving` so a modeled estimate and a real measurement
sit side by side. On success it records the result to the measurement
store (see below) — which **also stores the corresponding theoretical
roofline and the efficiency factor**, so each record documents reality
and theory together.

```
benchmark_serving(base_url, model, request_rate, input_len, output_len,
                  num_requests, max_concurrency=0, range_ratio=0.0,
                  endpoint="/v1/completions", gpu="",
                  tensor_parallel=1, pipeline_parallel=1, data_parallel=1,
                  expert_parallel=False, ignore_eos=True) -> report
```

- `base_url` is the server **root** (`http://host:8000`); a trailing
  `/v1` is stripped automatically.
- `model` is a single identifier — prefer a `PRESET_MODELS` key / HF id
  (e.g. `openai/gpt-oss-20b`). It is the tokenizer source *and* the key
  that lets the recorded theoretical baseline be computed. If the server
  serves under a different id (e.g. a local path), the tool auto-detects
  it from `/v1/models` and passes `--served-model-name` for you — the
  report's "Served as" line shows which id requests actually used.
- `request_rate` is the load knob (req/s, Poisson arrivals — mirrors
  `simulate_serving`). Pass `"inf"` to send everything at once
  (closed-loop, bounded by `max_concurrency`); in that mode the recorded
  theoretical baseline is skipped with a note, since the modeling
  simulator is open-loop.
- `max_concurrency` is an optional in-flight cap (default `0` = no cap —
  pure open-loop at `request_rate`). Concurrency is now a **result** of
  the run (observed peak in-flight), not an input.
- `gpu` is optional but recommended: it tags the recorded measurement so
  later lookups can calibrate by hardware, and (with a single-GPU,
  preset model+GPU and a finite `request_rate`) enables the stored
  theoretical baseline. Prefer a `PRESET_GPUS` name.
- `tensor_parallel` / `pipeline_parallel` / `data_parallel` /
  `expert_parallel` describe the **server's** parallelism layout (TP / PP
  / DP / EP). `vllm bench serve` is a *client* — it can't change how the
  server is sharded — so these are descriptive metadata: they're echoed
  in the report (total GPUs = TP×PP×DP) and stored on the measurement so
  results from different layouts aren't conflated on lookup. (The roofline
  baseline is only computed for single-GPU deployments; the modeling tools
  don't yet model TP/PP/DP scaling.)

Requires `vllm` installed and a reachable, already-running server. Also
runnable as a CLI:

```bash
uv run python -m agent.tools.benchmarking.benchmark \
    --base-url http://localhost:8000 --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --request-rate 10 --input-len 1024 --output-len 128 --num-requests 200 \
    --gpu h100-sxm
```

### Measurement store (`record_measurement` / `lookup_measurements`)

Successful benchmarks (and user-reported numbers) are persisted to a
shared, cross-session JSONL store under `$AGENT_MEASUREMENTS_DIR`
(default `measurements/`). Each record captures the measured metrics
**and the corresponding theoretical roofline** — computed from the same
preset model + GPU at the same operating point via the serving simulator
— plus the **efficiency factor** (fraction of the roofline achieved:
throughput = measured ÷ theory, latency = theory ÷ measured). So a single
record reads, e.g.:

```
gpt-oss-20b on 4090 | rate=5 req/s, c=11 (peak in-flight) in=256 out=64 tp=1
  measured: out=305 tok/s, TTFT=43ms, TPOT=7.4ms
  theory  : out=267 tok/s, TTFT=15ms, TPOT=4.4ms  [SATURATED in theory]
            | efficiency: output tput 114%, TPOT 60% of ideal
```

`lookup_measurements(model, gpu)` returns matching records so the agent
can reality-adjust a fresh estimate. Theory is computed only when the
model and GPU are exact preset names and the deployment is single-GPU
(the modeling tools don't yet model TP/PP/DP scaling); otherwise the
record stores a short note explaining why theory was skipped.

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

### Extend the benchmark tool

`benchmark_serving` lives at
[agent/tools/benchmarking/benchmark.py](agent/tools/benchmarking/benchmark.py)
and wraps `vllm bench serve` (subprocess) against a live endpoint. To add
a different load shape (e.g. a ShareGPT dataset, a concurrency sweep, or
a single-stream latency probe), add a sibling `@tool` that builds a
different `vllm bench serve` argument list and parses its `--save-result`
JSON via the same `ReportBuilder` and `record_measurement` plumbing.

### Add a new tool

Drop a `@tool`-decorated function under `agent/tools/` and import the
module from `agent/tools/__init__.py` (or one of its subpackages). The
argument schema is derived from the function signature + type hints.
See [agent/tools/base.py](agent/tools/base.py) for the registry contract.

## See also

- [SPEC.md](SPEC.md) — chassis contracts (loop, tool registry, config)
- [AGENTS.md](AGENTS.md) — coding style + repo conventions
