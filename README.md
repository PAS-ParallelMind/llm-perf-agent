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
  slash.py           shared slash-command dispatch (REPL + webui)
  types.py           SessionMeta dataclass
  workspace.py       session workspace (cwd for tools)
  tools/
    base.py          @tool registry + JSON-schema export
    fs.py            read / write / edit / glob / grep
    bash.py          sandboxed shell
    benchmarking/
      benchmark.py     benchmark_serving — MEASURED TTFT/TPOT/throughput
                       (wraps `vllm bench serve` against a live endpoint)
    modeling/
      memory.py      estimate_memory    — weights + KV cache VRAM breakdown
      latency.py     (plumbing)          — per-forward-pass latency model used by serving.py
      serving.py     simulate_serving   — continuous-batching workload sim
      report.py      ReportBuilder      — shared text-report helpers
      configs/
        hw_specs.py    PRESET_GPUS      — RTX 4090 / 5090, A100, H100, H200,
                                          B200, DGX Spark (GB10)
        hw_profiles/   per-GPU microbench grids (gitignored — measured)
        model_specs.py PRESET_MODELS    — gpt-oss-20b, Qwen3-Coder-30B (BF16 + AWQ-4bit)
    planning/
      evaluate_all.py  evaluate_all     — multi-candidate (gpu, tp, dp) sweep
                                          for the deployment-planning skill
    skills.py          list_skills / invoke_skill — pull a multi-step playbook
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
server with `vllm bench serve` and reports real TTFT / TPOT / throughput.
When the measurement diverges from the prediction, the agent saves a
brief note via `remember` (no separate measurement store).

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

Runs a Poisson-arrival workload (or `request_rate=.inf` for closed-loop)
through a vLLM-style continuous-batching scheduler with a
`max_num_batched_tokens` budget, accumulates per-phase latencies, and
reports TTFT / TPOT / E2E percentiles, observed concurrency, KV-cache
pressure, per-op step breakdown, and a saturation flag if the
deployment can't sustain the offered load.

```
simulate_serving(workload_file, gpu, tp=1, dp=1,
                 latency_source="baseline") -> report
```

- `workload_file` is a YAML carrying every workload knob (model,
  request_rate, input/output_len, max_num_batched_tokens, optional
  max_concurrent_requests / range_ratio / target_request_latency_s).
  Keeps simulator and benchmark apples-to-apples by reading the same
  file. `num_requests` is **not** a YAML knob — the tool auto-derives
  it from `request_rate` (`min(6000, max(200, 120 × rate))`; closed-
  loop fallback 500) so the agent can't get it wrong.
- `tp` / `dp` set the parallelism (tensor-parallel + data-parallel
  replicas). Total GPUs = tp × dp. TP shards heads + intermediate per
  layer and adds two ring all-reduces using the GPU's NVLink
  bandwidth; DP scales served rate and KV budget per replica.
- `latency_source` picks the per-op model:
  - `"baseline"` (default) — interpolates each op's wall time from
    the GPU's microbench grid (under
    [agent/tools/modeling/configs/hw_profiles/&lt;gpu&gt;/](agent/tools/modeling/configs/hw_profiles/)).
    The realistic projection; tracks measured TPOT within ~15% on
    B200 and ~13% on H200 across N = 1..256.
  - `"theoretical"` — analytic FLOPs/bytes vs theoretical peak
    (efficiency = 1.0 every op). The optimistic ceiling — over-
    predicts throughput by ~5–10× on real hardware. Use to scope
    optimisation headroom, not for projection.

Each modeling module is also runnable as a standalone CLI for ad-hoc
checks, e.g.:

```bash
uv run python -m agent.tools.modeling.memory \
    --model openai/gpt-oss-20b --concurrency 32 --context-length 4096

uv run python -m agent.tools.modeling.serving \
    --workload-file stages/01_workload.yaml --gpu h200-nvl \
    --tp 1 --dp 1 --latency-source baseline
```

### `evaluate_all`

For each GPU type the user has available, enumerates every valid
`(tp, dp)` split (with `tp*dp <= count` and `tp` dividing the model's
KV-head count), runs `simulate_serving` per cell, and returns a
cost-vs-latency Pareto table. The deployment-planning skill calls this
twice — once with `latency_source="baseline"`, once with
`"theoretical"` — to produce the realistic-projection and
analytic-ceiling tables side by side. Cost in `$/1M tok` is the
deployment cost (`tp × dp × per-GPU $/h`), so heavier configurations
are correctly penalised.

```
evaluate_all(workload_file, candidates, latency_source="baseline") -> report
# candidates: [{"gpu": "<PRESET_GPUS key>", "count": <int>}, ...]
```

## Measured benchmarking tool

### `benchmark_serving`

The measured counterpart to `simulate_serving`. Drives a synthetic
`random` workload through a **running** OpenAI-compatible server (e.g.
vLLM) with `vllm bench serve` and reports real TTFT / TPOT / ITL / E2EL
(mean / median / p99) plus request and token throughput. The parameters
mirror `simulate_serving` so a modeled estimate and a real measurement
sit side by side. There is no measurement store; cross-session
continuity comes from `remember` notes the agent saves when measured
diverges from predicted (see "Tracking prediction drift" below).

```
benchmark_serving(base_url, workload_file, endpoint="/v1/completions",
                  gpu="",
                  tensor_parallel=1, pipeline_parallel=1, data_parallel=1,
                  expert_parallel=False, ignore_eos=True) -> report
```

- `base_url` is the server **root** (`http://host:8000`); a trailing
  `/v1` is stripped automatically.
- `workload_file` is the same YAML shape `simulate_serving` consumes
  (model, request_rate, input/output_len, optional max_concurrent_requests
  / range_ratio). `num_requests` is auto-derived from `request_rate`
  — don't put it in the YAML. Pass `request_rate: inf` for closed-loop
  mode.
- `model` is read from the YAML — prefer a `PRESET_MODELS` key / HF id
  (e.g. `openai/gpt-oss-20b`). It is the tokenizer source. If the server
  serves under a different id (e.g. a local path), the tool auto-detects
  it from `/v1/models` and passes `--served-model-name` for you — the
  report's "Served as" line shows which id requests actually used.
- `gpu` labels the measurement in the report header; prefer a
  `PRESET_GPUS` name. Blank = "unknown".
- `tensor_parallel` / `pipeline_parallel` / `data_parallel` /
  `expert_parallel` describe the **server's** parallelism layout (TP / PP
  / DP / EP). `vllm bench serve` is a *client* — it can't change how the
  server is sharded — so these are descriptive metadata: they're echoed
  in the report (total GPUs = TP×PP×DP). PP is still unmodelled and
  any `pipeline_parallel != 1` makes drift-comparison against the
  simulator meaningless.

Requires `vllm` installed and a reachable, already-running server. Also
runnable as a CLI:

```bash
uv run python -m agent.tools.benchmarking.benchmark \
    --base-url http://localhost:8000 \
    --workload-file stages/01_workload.yaml \
    --gpu h100-sxm
```

### Tracking prediction drift

The agent doesn't maintain a measurement store. Instead, when
`benchmark_serving` produces a number that diverges meaningfully from
the matching `simulate_serving(latency_source="baseline")` prediction
(rule of thumb: > 20% TPOT gap), the agent saves a one-line
observation via `remember`. Those notes are loaded automatically into
the system prompt on future sessions, so the agent has cross-session
awareness of where the microbench grid for a given `(model, gpu)`
pair may be stale — without needing a separate JSONL store.

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

Slash commands inside the REPL (and the webui input box):

```
/help      list commands
/tools     list registered tools
/plan      load the deployment_planning playbook (activates planning mode)
/reset     clear conversation history (keep the session dir)
/exit      quit (REPL only)
```

`/plan` is the user-facing way to enter planning mode — the slash
command pre-loads the playbook into the conversation, after which
the agent walks through the four-stage deployment workflow. The
agent doesn't invoke this on its own; you opt in explicitly.

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
JSON via the same `ReportBuilder` plumbing.

### Add a new tool

Drop a `@tool`-decorated function under `agent/tools/` and import the
module from `agent/tools/__init__.py` (or one of its subpackages). The
argument schema is derived from the function signature + type hints.
See [agent/tools/base.py](agent/tools/base.py) for the registry contract.

## See also

- [SPEC.md](SPEC.md) — chassis contracts (loop, tool registry, config)
- [AGENTS.md](AGENTS.md) — coding style + repo conventions
