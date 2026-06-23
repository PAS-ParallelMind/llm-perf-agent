# Agent for LLM Inference Performance — Specification

Reference for the chassis: what guarantees it makes, and where to extend
it. Read [README.md](README.md) for tutorial-style onboarding; read this
for contracts.

---

## 1. Purpose

A multi-turn chat agent driving an LLM — via an OpenAI-compatible
tool-calling endpoint — that helps users with **LLM inference deployment
guidance and performance analysis**. The agent calls performance tools
on the user's behalf, interprets results, and answers follow-ups across
turns.

It is **not** a benchmark runner, a serving framework, or a hardware
oracle. It is glue between:

- a vLLM-served (or any OpenAI-compatible) chat model,
- a tool-calling loop with filesystem / shell / perf-analysis tools,
- a REPL that persists messages, exports traces, and survives crashes.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ agent/main.py             REPL + session bookkeeping         │
├──────────────────────────────────────────────────────────────┤
│ agent/loop.py             ChatAgent — multi-turn loop        │
│ agent/engine.py           OpenAI chat.completions wrapper    │
│ agent/tools/*             FS / shell / perf tools            │
│ agent/memory.py           persistent cross-session notes     │
├──────────────────────────────────────────────────────────────┤
│ agent/workspace.py        thread-local session workspace     │
└──────────────────────────────────────────────────────────────┘
```

**Strict layering**: a layer calls only downward. Tools never reach into
main.py; the loop never imports the REPL. The loop is domain-agnostic —
the perf focus comes from the registered tools + the system prompt.

---

## 3. Core types

### `ChatAgent` (agent/loop.py)
Holds:
- `messages`: persistent OpenAI-format conversation
- `tool_call_log`: append-only log of every dispatch (incl. `<llm>`)
- `tool_call_counts`: anti-thrash counter keyed by `(name, args)`
- `turn_count`, `max_steps`

Public methods: `chat(user_message) -> TurnResult`, `reset()`.

### `TurnResult`
```python
reply: str             # final assistant text (what to show the user)
steps: int             # tool-call iterations consumed this turn
elapsed_s: float
tool_calls: list[dict] # this turn's dispatch log
truncated: bool        # True if max_steps hit before a reply
```

### `SessionMeta` (agent/types.py)
Top-level info written to `<session>/batch/session/summary.json` after
every turn.

---

## 4. Tool contract

Tools live in `agent/tools/*` and register themselves via the `@tool`
decorator (`agent/tools/base.py`). The decorator:

1. reads the function signature + type hints,
2. builds a JSON-schema parameter object,
3. appends the function to a module-level registry keyed by name.

`ChatAgent.chat()` calls `tools.base.schemas()` for the LLM's tool list
and `tools.base.dispatch(name, args)` to execute each call.

**Invariants:**

- Every tool returns a single string. Truncation happens at the loop
  boundary (16 KB, in `agent/loop.py`), not inside the tool.
- Tools are pure Python callables — no async, no generators.
- File-modifying tools (`write_file`, `edit_file`, …) operate within the
  session workspace; paths resolve via `agent.workspace.resolve()` which
  blocks `..` escapes.
- `remember` / `recall` are the only tools allowed to write outside the
  workspace (they target `memory/`).
- `read_file` is paginated (`offset` + `limit`, default 200 lines).

**Registered tools today:**

| Tool                | Module                              | Purpose                                                |
|---------------------|-------------------------------------|--------------------------------------------------------|
| `read_file`         | tools/fs.py                         | Paginated read (offset/limit, line-numbered)           |
| `write_file`        | tools/fs.py                         | Overwrite / create                                     |
| `edit_file`         | tools/fs.py                         | String-replace (`old` → `new`)                         |
| `glob`              | tools/fs.py                         | List matching paths                                    |
| `grep`              | tools/fs.py                         | Regex across workspace                                 |
| `bash`              | tools/bash.py                       | Shell exec, timeout-bounded                            |
| `benchmark_serving` | tools/benchmarking/benchmark.py     | MEASURED serving probe: wraps `vllm bench serve` → TTFT/TPOT/throughput |
| `record_measurement`| tools/benchmarking/measurements.py  | Persist a measured result to the cross-session store    |
| `lookup_measurements`| tools/benchmarking/measurements.py | Read back measured results to calibrate estimates       |
| `estimate_memory`   | tools/modeling/memory.py            | VRAM breakdown: weights + KV cache (per model/concurrency/context) |
| `simulate_serving`  | tools/modeling/serving.py           | Continuous-batching workload sim: TTFT / TPOT / throughput. `latency_source`: `baseline` (microbench-calibrated, default) or `theoretical` (analytic roofline) |
| `remember`          | memory.py                           | Save a memory file + index entry                       |
| `recall`            | memory.py                           | Read a memory file by name                             |

The modeling tools take structured kwargs (model + GPU preset
names from `tools/modeling/configs/`, plus workload-shape ints) and
return a formatted text report rendered via `tools/modeling/report.py`.
`benchmark_serving` is the measured counterpart to `simulate_serving`:
it shells out to `vllm bench serve` (keeping vLLM's torch/CUDA import out
of the agent process), parses the `--save-result` JSON, renders via the
same `ReportBuilder`, and on success records the result through
`record_measurement` so `lookup_measurements` can calibrate later
estimates. It needs `vllm` installed and a reachable running server, and
takes real wall-clock time — the loop's per-turn anti-thrash cap keeps it
from being re-fired blindly.

---

## 5. Loop contract

`ChatAgent.chat(user_message)` runs one user turn:

1. Refresh the system prompt (picks up memory edits made mid-session).
2. Append the user message; tick `turn_count`.
3. Up to `max_steps` iterations of:
   a. `engine.chat(messages, tools=schemas)`
   b. If `reasoning=True`, prepend the reasoning to `content`.
   c. **No tool calls** → that text is the final reply; return.
   d. Otherwise: dispatch each tool (sequentially), append result as
      `role:"tool"`, log to `tool_call_log`.
4. If `max_steps` exhausted, return `truncated=True` with a sentinel
   reply.

Anti-thrash: `_MAX_TOOL_CALLS_PER_TURN=4` (dedupe within a turn),
`_MAX_IDENTICAL_TOOL_CALLS=3` (per session, across turns).

The loop never raises to its caller; engine errors propagate up so the
REPL can show them without killing the session.

---

## 6. Configuration model

A single YAML, schema in `agent/config.py::ChatConfig`:

```yaml
agent:
  model:
    name:        <str>
    base_url:    <str>
    api_key:     <str>
    temperature: <float|null>
    max_tokens:  <int>
    reasoning:   <bool>
  max_steps:     <int>

session:
  dir:           <path>      # parent dir for session subdirs
  name:          <str|null>  # auto chat-YYYYMMDD-HHMMSS if null

# One of:
system_prompt:      |
  <multi-line task-specific guidance>
system_prompt_file: <path>
```

Inline `system_prompt` wins over `system_prompt_file` when both are set.

The hardware / model *under analysis* is **not** in config. Tools take
their own arguments per call.

---

## 7. Session output layout

```
<session.dir>/<session.name>/
  run.yaml                              config snapshot
  batch/session/                        per-session workspace
    trace.json                          full message history
    tool_calls.jsonl                    step-indexed tool log
    summary.json                        SessionMeta
    *                                   whatever the agent wrote
```

The nesting (`batch/session/`) is for compatibility with the legacy
webui under `webui/`. After every user turn, the loop re-serializes
trace.json + tool_calls.jsonl + summary.json — partial sessions remain
inspectable.

---

## 8. Memory

`memory/` stores persistent notes usable across sessions.

- `MEMORY.md` is the index: `- [title](file.md) — hook` per line.
- Each memory file has YAML frontmatter (`name`, `description`, `type`).
- `remember` writes both the file and the index line.
- `recall` reads a memory file by filename.
- `prompts.build_system_prompt()` always injects the index under a
  `## Memory` section.

Memory is **optional** and **inspectable** — nothing in the loop
requires an entry to exist.

---

## 9. Invariants and non-goals

### Invariants
1. **No domain-specific logic in the loop**. Domain focus comes from
   registered tools + the system prompt.
2. **Workspace isolation**: each session's filesystem effects are
   confined to its `batch/session/` directory.
3. **Crash-safe trace**: after every user turn, the full trace is
   re-serialized.
4. **No hidden global state**: the workspace root is thread-local; the
   tool registry is process-wide but populated declaratively at import.

### Non-goals
- **No batch mode**: this is an interactive chat agent. For headless
  evaluation, drive `ChatAgent.chat()` from a Python script.
- **No model-response caching**: every turn re-issues every prompt.
- **No live endpoint management**: the agent does not spawn or tear
  down inference servers. If a tool needs to talk to one, the URL is
  passed as a tool argument.

---

## 10. Extending — adding a tool

1. Create `agent/tools/<your_tool>.py`, or drop a module into an
   existing subpackage (e.g. `agent/tools/modeling/<your_tool>.py`,
   `agent/tools/benchmarking/<your_tool>.py`) when the tool fits a
   thematic group. Subpackages are imported by `agent/tools/__init__.py`
   via their own `__init__.py`, which side-effect imports the leaf
   modules so the `@tool` decorator runs.
2. Decorate a regular function with `@tool(description, **param_desc)`.
   Type-hint every parameter — those types drive the JSON schema.
3. Register the module: either add `from . import <your_tool>` to
   `agent/tools/__init__.py` for top-level tools, or to the subpackage's
   own `__init__.py` for grouped tools.
4. (Optional) Update `agent/prompts.py` so the default system prompt
   tells the model when to reach for it.

Default values become optional parameters. Return value must be a single
string; long results are auto-truncated at 16 KB. For tools that produce
multi-section reports, prefer composing the output via
`agent/tools/modeling/report.py::ReportBuilder` so formatting stays
consistent across tools.
