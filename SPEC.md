# ParallelMind Harness — Specification

Reference for what this framework **is**, what guarantees it makes, and
what its extension points are. Read `README.md` for tutorial-style
onboarding; read this document to understand the contracts.

---

## 1. Purpose

ParallelMind Harness drives an LLM — via an OpenAI-compatible
tool-calling endpoint — through pre-rendered programming tasks, runs
each task in a sandboxed workspace, and emits a unified results JSON.

It is **not** a parallel runtime, compiler, scheduler, or benchmark
evaluator. It is glue between:

- a vLLM-served model speaking OpenAI chat-completions,
- a tool-calling loop with filesystem / shell / build tools,
- a unified JSON-in/JSON-out contract that a benchmark-specific
  preprocessor produces.

The harness itself is benchmark-agnostic. Suite-specific code (ParEval,
HeCBench, ParallelMind eval, …) lives outside the harness as
preprocessing scripts that emit `problems.json` and downstream
evaluators that consume the harness's output.

---

## 2. Architecture

Three layers, top → bottom:

```
┌──────────────────────────────────────────────────────────────┐
│ agent/batch.py            config-driven runner + concurrency │
├──────────────────────────────────────────────────────────────┤
│ agent/loop.py             tool-calling agent loop            │
│ agent/engine.py           OpenAI chat.completions wrapper    │
│ agent/tools/*             FS / shell / build / submit tools  │
│ agent/memory.py           persistent cross-run notes         │
├──────────────────────────────────────────────────────────────┤
│ agent/workspace.py        thread-local per-task working dir  │
│ agent/submission.py       thread-local submit_solution sink  │
└──────────────────────────────────────────────────────────────┘
```

**Strict layering**: a layer calls only downward. Tools never reach
into batch.py, the loop never imports the batch runner. The loop is
benchmark-agnostic and the tools are reusable across runs.

---

## 3. Core data types

Defined in `agent/types.py`. Both are `@dataclass` with no methods —
pure data envelopes.

### `AgentTask`
```python
id: str                          # unique problem identifier
instruction: str                 # natural-language prompt for the agent
metadata: dict[str, Any]         # pass-through — copied to AgentResult
```

- `id` doubles as the per-task workspace directory name, so it must be
  filesystem-safe.
- `instruction` is the first `user` message. The system prompt comes
  from `RunConfig.system_prompt` + the memory index.
- `metadata` is preserved untouched; output JSON copies it back.

### `AgentResult`
```python
task_id: str
code: str                        # submitted (or fallback-extracted) code
raw_reply: str                   # the model's final text reply
trace: list[dict]                # full message history
tool_calls: list[dict]           # step-indexed tool-call log
steps: int
elapsed_s: float
submitted: bool                  # True iff submit_solution was called
error: str | None
metadata: dict                   # copy of AgentTask.metadata + harness-internal additions
```

If the agent never calls `submit_solution`, `batch.py` falls back to
extracting the last fenced code block from `raw_reply`; `submitted`
stays False.

---

## 4. Input / output JSON contract

The harness's only external contract: read `problems.json`, write a
results JSON. No template substitution, no benchmark-specific routing.

### Input — `problems.json`

```json
[
  {
    "id":          "P001",
    "prompt":      "...already-rendered text...",
    "seed_files":  { "<relpath>": "<content>", ... },   // optional
    "metadata":    { ... }                              // optional
  }
]
```

**Invariants:**

1. `id` values are unique within a file. The harness rejects
   duplicates at load time.
2. `prompt` is the verbatim user message — the harness performs no
   `{{var}}` substitution. The producing script is responsible for
   rendering whatever template it likes.
3. `seed_files` keys are forward-slash relative paths. Absolute paths
   and `..` segments are rejected. Nested directories are created
   automatically. Each file is written into the per-task workspace
   before the agent loop starts.
4. `metadata` is copied verbatim into the matching output entry. The
   harness never reads its keys (one exception: a harness-internal
   `workspace` key is added during execution and stripped before
   export).

### Output

`<output>.json`:
```json
[
  {
    "id":         "P001",
    "code":       "...submitted source...",
    "submitted":  true,
    "steps":      21,
    "elapsed_s":  356.0,
    "error":      null,
    "metadata":   { ... passthrough ... }     // omitted if empty
  }
]
```

`<output_stem>.code.json` (always written):
```json
[ {"id": "P001", "code": "..."}, ... ]
```
Only entries where `submitted=True` and `code` is non-empty appear.
Convenient for piping into a downstream evaluator.

**Crash-safe incremental export**: after each task completes,
`batch.py` re-serializes the full partial result set. A crash mid-run
leaves a valid (partial) output JSON.

---

## 5. Tool contract

Tools live in `agent/tools/*` and register themselves via the `@tool`
decorator (`agent/tools/base.py`). The decorator:

1. reads the function signature + type hints,
2. builds a JSON-schema parameter object,
3. appends the function to a module-level registry keyed by name.

`Agent.run()` calls `tools.base.schemas()` once to get the schema list
for the LLM, and `tools.base.dispatch(name, args)` to execute each
tool call.

**Invariants:**

- Every tool returns a single string. Truncation happens at the loop
  boundary (16 KB, in `agent/loop.py`), not in the tool.
- Tools are pure Python callables — no async, no generators.
- File-modifying tools (`write_file`, `edit_file`, `omp_build_and_run`,
  …) operate within the current workspace root; paths resolve via
  `agent.workspace.resolve()` which blocks escaping via `..`.
- `submit_solution` is the only tool that terminates the loop. It
  sets a thread-local value read by the loop at end-of-turn.
- `remember` / `recall` are the only tools allowed to write outside
  the workspace (they target `memory/`).
- **`read_file` is paginated** (`offset` + `limit`, default 200
  lines). Responses start with a `[lines X–Y of N]` header so the
  agent knows where to continue.

Available tools today:

| Tool                   | Module              | Purpose                                     |
|------------------------|---------------------|---------------------------------------------|
| `read_file`            | tools/fs.py         | Paginated read (offset/limit, line-numbered) |
| `write_file`           | tools/fs.py         | Overwrite / create                          |
| `edit_file`            | tools/fs.py         | String-replace (`old` → `new`)              |
| `glob`                 | tools/fs.py         | List matching paths                         |
| `grep`                 | tools/fs.py         | Regex across workspace                      |
| `bash`                 | tools/bash.py       | Shell exec, timeout-bounded                 |
| `omp_build_and_run`    | tools/parallel.py   | gcc/clang/icpx + run                        |
| `nvcc_build_and_run`   | tools/parallel.py   | nvcc + run                                  |
| `mpi_build_and_run`    | tools/parallel.py   | mpicc/mpicxx + `mpirun -np N` + run         |
| `hardware_info`        | tools/parallel.py   | Probe GPUs / nvcc / host compilers          |
| `submit_solution`      | tools/submit.py     | Submit final code and stop the loop         |
| `remember`             | memory.py           | Write a memory file + index entry           |
| `recall`               | memory.py           | Read a memory file by name                  |

---

## 6. Agent loop contract

`agent/loop.py::Agent.run(task, time_budget_s)` is the only place the
LLM is invoked. One turn =

1. Call `engine.chat(messages, tools=schemas)`.
2. Append assistant message (with `tool_calls` if any).
3. If `reasoning=True` on the engine, prepend the reasoning string to
   `content` so it's echoed on the next turn.
4. If no tool calls: send an idle-nudge; after `_MAX_IDLE_TURNS=3`
   consecutive idle turns, stop.
5. Otherwise, dispatch each tool call sequentially, append the result
   as a `tool` message, log to `tool_call_log`.
6. If `submission.get() is not None`, the loop exits successfully.

Exits: successful submission, time budget exceeded, `max_steps`
reached, or idle-streak exceeded. Every exit produces an `AgentResult`
— the loop never raises to its caller except for engine / network
errors, which `batch.py` catches and records as `error`.

---

## 7. Configuration model

A single YAML drives a run. Schema in `agent/config.py::RunConfig`:

```yaml
model:
  name:        <str>     # e.g. openai/Qwen3-Coder-30B-A3B-Instruct
  base_url:    <str>     # vLLM OpenAI endpoint
  api_key:     <str>     # usually "EMPTY"
  temperature: <float>
  max_tokens:  <int>
  reasoning:   <bool>    # echo model's chain-of-thought back each turn

agent:
  max_steps:   <int>     # hard cap on loop iterations
  time_budget: <int>     # per-task wall-clock seconds
  workers:     <int>     # concurrent tasks (threads)

io:
  input:           <abs path to problems.json>
  output:          <abs path to results JSON>
  workspace_root:  <abs path; per-task subdirs created here>

# One of:
system_prompt: |
  <multiline task-specific guidance>
# or
system_prompt_file: <path>
```

The system prompt is concatenated with the auto-loaded memory index
(`memory/MEMORY.md`) to produce the final system message. If both
`system_prompt` and `system_prompt_file` are set, the inline string
wins.

---

## 8. Run output layout

Everything for a run lives under whatever directory you choose; the
harness only manages `io.output` and `io.workspace_root`:

```
<run-dir>/
  run.yaml                         # input: config
  problems.json                    # input: pre-rendered prompts
  agent_output.json                # output: per-task results
  agent_output.code.json           # auxiliary: [{id, code}, ...]
  batch/<task_id>/                 # per-task workspace
    trace.json                     # full message history
    tool_calls.jsonl               # step-indexed tool log
    summary.json                   # steps / elapsed / submitted / error
    *.cu, *.cpp, a.out             # whatever the agent wrote
```

Filenames inside `batch/<task_id>/` aren't enforced by the harness —
the agent can write anything. `trace.json` / `tool_calls.jsonl` /
`summary.json` are produced by `batch.py` after the loop finishes.

---

## 9. Memory system

`memory/` stores persistent notes usable across runs.

- `MEMORY.md` is the index — one line per memory:
  `- [title](file.md) — hook`.
- Per-memory files have YAML frontmatter (`name`, `description`,
  `type`).
- The `remember` tool writes both the file and the index line
  atomically.
- The `recall` tool reads a memory file by filename.
- `prompts.build_system_prompt()` always loads the index and injects
  it under a `## Memory` section in the system message.

Memory is **optional** and **inspectable** — nothing in the loop
requires a memory entry to exist. The index is truncated to fit the
context window.

---

## 10. Invariants and non-goals

### Invariants
1. **No benchmark-specific logic anywhere in the harness**. All
   benchmark-aware logic lives in external preprocessing scripts that
   emit `problems.json`.
2. **Workspace isolation**: each task's filesystem effects are
   confined to its `batch/<task_id>/` directory. Tool dispatch
   resolves paths through `workspace.resolve()` which rejects `..`
   escapes.
3. **Thread safety via thread-locals**: `workspace.set_root()` and
   `submission.reset()` are thread-local, so `workers: N` in config
   produces N independent agent runs without mutex.
4. **Deterministic re-runs**: with `temperature: 0.0`, the same
   `problems.json` produces near-identical output across runs. (Some
   non-determinism remains under concurrent vLLM load; known quirk.)
5. **Crash-safe incremental export**: after each task completes,
   `batch.py` re-serializes the full partial result set.

### Non-goals
- **No distributed scheduling**: workers are threads in one process.
  For multi-host runs, split `problems.json` and launch N instances.
- **No sandboxing beyond `subprocess` + `cwd`**: the `bash` tool can
  read anything the process can read. Run in a constrained container
  for untrusted models.
- **No template substitution**: the harness will not interpret
  `{{var}}` or any other syntax in `prompt`. Render externally.
- **No caching of model responses**: every run re-issues every
  prompt.
- **No benchmark evaluator integration**: validating correctness or
  measuring performance is a downstream concern. The harness emits
  `code`; the consumer evaluates.

---

## 11. Extension — running a new benchmark

There is no harness-side change required. Write a small preprocessing
script that walks the benchmark's source tree and emits a
`problems.json`. For each problem:

1. Render whatever prompt / instructions you want into `prompt`.
2. If the task needs scaffolding files (headers, datasets, reference
   sources), inline them into `seed_files`.
3. Put any downstream-consumer-relevant fields (category, expected
   validation type, parallelism model, ...) into `metadata`.

Then write a `run.yaml` pointing at the output and run
`python -m agent.batch --config run.yaml`. The result JSON has
everything a downstream evaluator needs (id, code, metadata
passthrough, submitted flag).

`eval/build_problems_json.py` (in this repo) is a 30-line example
that converts ParallelMind's own benchmarks into the harness format.
