# ParallelMind Harness — Specification

Reference for what this framework **is**, what guarantees it makes, and what
its extension points are. Read `README.md` for tutorial-style onboarding; read
this document to understand the contracts.

---

## 1. Purpose

ParallelMind Harness drives an LLM — via an OpenAI-compatible tool-calling
endpoint — through benchmark-defined parallel programming tasks, then feeds
each generated solution into the benchmark's own evaluator to produce
correctness + performance numbers.

It is **not** a parallel runtime, compiler, or scheduler. It is the glue
between:

* a vLLM-served model speaking OpenAI chat-completions,
* a tool-calling loop with filesystem / shell / build tools,
* per-benchmark adapters that translate between the benchmark's format and
  the agent's internal format,
* driver scripts that orchestrate generate → evaluate → metrics.

Supported benchmarks today: **ParEval** (function-level prompts, 60 problems
across OMP / MPI / CUDA / Kokkos / HIP / serial) and **HeCBench** (~500
full-program heterogeneous benchmarks). Extending to a new suite is a
~200-line adapter + a run script.

---

## 2. Architecture

Four layers, top → bottom:

```
┌──────────────────────────────────────────────────────────────┐
│ scripts/run_<bench>.py    orchestration: agent → eval → metrics│
├──────────────────────────────────────────────────────────────┤
│ agent/batch.py            per-task runner + concurrency       │
│ agent/adapters/<name>.py  benchmark ↔ AgentTask/AgentResult   │
├──────────────────────────────────────────────────────────────┤
│ agent/loop.py             tool-calling agent loop             │
│ agent/engine.py           OpenAI chat.completions wrapper     │
│ agent/tools/*             FS / shell / build / submit tools   │
│ agent/memory.py           persistent cross-run notes          │
├──────────────────────────────────────────────────────────────┤
│ agent/workspace.py        thread-local per-task working dir   │
│ agent/submission.py       thread-local submit_solution sink   │
└──────────────────────────────────────────────────────────────┘
```

**Strict layering**: a layer calls only downward. Adapters never import
`loop.py`; tools never import adapters. This keeps the loop benchmark-agnostic
and the tools reusable across benchmarks.

---

## 3. Core data types

Defined in `agent/adapters/base.py`. Both are `@dataclass` with no methods —
they are pure data envelopes.

### `AgentTask`
```python
id: str                          # unique problem identifier
instruction: str                 # natural-language prompt for the agent
metadata: dict[str, Any]         # pass-through — returned verbatim in AgentResult
```

* `id` doubles as the per-task workspace directory name, so it must be
  filesystem-safe.
* `instruction` is appended as the first `user` message. The system prompt
  comes from `AgentConfig.system_prompt` + the memory index.
* `metadata` is preserved untouched so `export()` can reconstruct the
  benchmark-native record.

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
metadata: dict                   # copy of AgentTask.metadata
```

If the agent never calls `submit_solution`, `batch.py` falls back to
extracting the last fenced code block from `raw_reply`; `submitted` stays
False.

---

## 4. BenchmarkAdapter contract

```python
class BenchmarkAdapter(ABC):
    def load(self, path: str, **kwargs) -> list[AgentTask]: ...
    def export(self, results: list[AgentResult], output_path: str) -> None: ...
```

**Invariants:**

1. `load()` is pure — no network, no mutation of input files.
2. For every task `t` produced by `load()`, `t.metadata` contains
   enough information for `export()` to reproduce the benchmark-native
   record without re-reading the benchmark repo.
3. `export()` accepts results in any order; the output format must be
   self-describing (including task ids / names) so it can be consumed by the
   benchmark's native evaluator.
4. Any normalization of model output (stripping fences, signatures, etc.)
   happens in `export()`, not in `loop.py`. The adapter is the only layer
   that knows what the benchmark's evaluator accepts.

Registered in `agent/batch.py::ADAPTERS` — a string key routes
`--adapter <name>` on the batch CLI.

**Optional workspace pre-seeding.** An adapter may set
`task.metadata["seed_dir"] = "<path>"`. Before the agent loop
starts, `batch.py` copies every file in that directory into the
per-task workspace (shallow, skips dotfiles). Lets the agent call
build tools against headers / reference files without needing a
`write_file` per dependency. HeCBench uses this to ship
`reference.h` alongside the agent's main source.

---

## 5. Tool contract

Tools live in `agent/tools/*` and register themselves via the `@tool`
decorator (`agent/tools/base.py`). The decorator:

1. reads the function signature + type hints,
2. builds a JSON-schema parameter object,
3. appends the function to a module-level registry keyed by name.

`Agent.run()` calls `tools.base.schemas()` once to get the schema list for
the LLM, and `tools.base.dispatch(name, args)` to execute each tool call.

**Invariants:**

* Every tool has a single-string return value. Truncation happens at the
  loop boundary (16 KB, in `agent/loop.py`), not in the tool.
* Tools are pure Python callables — no async, no generators.
* File-modifying tools (`write_file`, `edit_file`, `omp_build_and_run`,
  etc.) operate within the current workspace root; paths resolve via
  `agent.workspace.resolve()` which blocks escaping via `..`.
* `submit_solution` is the only tool that terminates the loop. It sets a
  thread-local value read by the loop at end-of-turn.
* `remember` / `recall` are the only tools allowed to write outside the
  workspace (they target `memory/`).
* **`read_file` is paginated** (`offset` + `limit`, default 200 lines).
  Responses start with a `[lines X–Y of N]` header so the agent knows
  where to continue. For files whose content exceeds the 16 KB cap even
  after this, the agent re-reads with a smaller `limit`.

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

`agent/loop.py::Agent.run(task, time_budget_s)` is the only place the LLM
is invoked. One turn =

1. Call `engine.chat(messages, tools=schemas)`.
2. Append assistant message (with `tool_calls` if any).
3. If `reasoning=True` on the engine, prepend the reasoning string to
   `content` so it's echoed on the next turn.
4. If no tool calls: send an idle-nudge; after `_MAX_IDLE_TURNS=3`
   consecutive idle turns, stop.
5. Otherwise, dispatch each tool call sequentially, append the result as a
   `tool` message, log to `tool_call_log`.
6. If `submission.get() is not None`, the loop exits successfully.

Exits: successful submission, time budget exceeded, `max_steps` reached,
or idle-streak exceeded. Every exit produces an `AgentResult` — the loop
never raises to its caller except for engine / network errors, which
`batch.py` catches and records as `error`.

---

## 7. Configuration model

Two YAML files per run, under `runs/<bench>/<run-name>/`:

### `agent.yaml` — shared across benchmarks
Dataclasses in `agent/config.py::AgentConfig`:

```yaml
model:
  name:        <str>     # e.g. openai/gpt-oss-120b
  base_url:    <str>     # vLLM OpenAI endpoint
  api_key:     <str>     # usually "EMPTY"
  temperature: <float>
  max_tokens:  <int>
  reasoning:   <bool>    # echo model's chain-of-thought back each turn
agent:
  max_steps:   <int>     # hard cap on loop iterations
  time_budget: <int>     # per-task wall-clock seconds
  workers:    <int>     # concurrent tasks
system_prompt: |
  <multiline task-specific guidance>
```

The system prompt is concatenated with the auto-loaded memory index
(`memory/MEMORY.md`) to produce the final system message.

### `config.yaml` — benchmark-specific
One dataclass per benchmark. Current entries:

* `ParevalBenchmarkConfig` — `problem_set` (omp / mpi / cuda / ...),
  `launch_configs`, `build_timeout`, `run_timeout`.
* `HeCBenchBenchmarkConfig` — `target` (the parallel model to
  produce: cuda / omp / hip / sycl), `serial_root` (dir of
  pre-generated serial C++ sources), `src_root` (HeCBench `src/`
  tree used for the reference baseline), `names`, `categories`
  (optional filters), `repeat` (autohecbench `-r`), `nvidia_sm`
  (compute capability for autohecbench baseline timing, e.g. 89 for
  RTX 4090 — not seen by the agent, which probes the GPU itself via
  the `hardware_info` tool).

Adding a benchmark = add another dataclass + `from_yaml` classmethod.

---

## 8. Run output layout

Runs are grouped per-benchmark under `runs/<bench>/<run-name>/`, so
ParEval and HeCBench artifacts don't mix:

```
runs/
  pareval/
    _shared/                     # shared inputs (launch-configs, ...)
    _analysis/                   # cross-run aggregate CSVs/MDs
    <run-name>/
      agent.yaml                 # input: agent/model config
      config.yaml                # input: benchmark config
      system_prompt.txt          # derived: system_prompt excerpted from agent.yaml
      agent_output.json          # stage 1 output (adapter-normalized)
      results.json / results.csv / metrics.csv
      batch/<task_id>/           # per-task workspace
        trace.json               # full message history
        tool_calls.jsonl         # step-indexed tool log
        summary.json             # steps / elapsed / submitted / error
        solution.cpp             # canonical TU evaluated by the benchmark
        *.cpp, a.out             # agent scratch files
      scratch/                   # ParEval's own eval scratch
  hecbench/
    <run-name>/
      agent.yaml / config.yaml / system_prompt.txt / agent_output.json
      batch/<name>/              # per-task workspace, pre-seeded with
                                 # serial main.cpp + reference.h
      scratch/<name>-<target>/   # mirrored src dir with agent's candidate
                                 # main source substituted (Stage 2)
      baseline.csv               # Stage 3a: autohecbench timings on src
      candidate.csv              # Stage 3b: autohecbench timings on scratch
      speedup.md                 # Stage 4: autohecbench-compare output
      results.json / results.csv # Stage 4: merged per-task records
```

The `solution.cpp` in each `batch/<task_id>/` (ParEval) is guaranteed
to be the exact TU the benchmark compiled (`prompt + normalized
output`). For HeCBench, the evaluated source lives at
`scratch/<name>-<target>/main.cu` (or `main.cpp`) — each benchmark is
compiled and run by `autohecbench.py` via its own Makefile.

---

## 9. Memory system

`memory/` stores persistent notes usable across runs.

* `MEMORY.md` is the index — one line per memory: `- [title](file.md) — hook`.
* Per-memory files have YAML frontmatter (`name`, `description`, `type`).
* The `remember` tool writes both the file and the index line atomically.
* The `recall` tool reads a memory file by filename.
* `prompts.build_system_prompt()` always loads the index and injects it
  under a `## Memory` section in the system message.

Memory is **optional** and **inspectable** — nothing in the loop requires a
memory entry to exist. The index is truncated to fit the context window.

---

## 10. Scripts

### `scripts/run_pareval.py`
Three-stage pipeline: agent (via `agent.batch`) → ParEval eval
(`drivers/run-all.py`) → metrics (`analysis/metrics.py`). Writes
`solution.cpp` in each task dir after stage 1 so evaluated TU is
always inspectable.

### `scripts/run_bare.py`
Non-agent baseline that wraps ParEval's own `generate-openai-vllm.py`.
Output is post-processed through the adapter's `normalize()` so numbers
are apples-to-apples with agent mode.

### `scripts/run_hecbench.py`
Four-stage pipeline:

1. **Agent** via `agent.batch` with the HeCBench adapter — produces
   `agent_output.json`.
2. **Scratch tree** — mirrors `src/<name>-<target>/` →
   `scratch/<name>-<target>/` per submitted entry, overwriting the
   main source with the agent's candidate. Original HeCBench tree
   is never modified.
3. **Timing** — invokes `benchmarks/HeCBench/src/scripts/autohecbench.py`
   twice (against `src/` and against `scratch/`) to produce
   `baseline.csv` and `candidate.csv`. Each row has `config.repeat`
   timing samples.
4. **Compare** — runs `autohecbench-compare.py`, writes
   `speedup.md` + merged `results.json` / `results.csv`.

Delegating to HeCBench's own scripts (instead of reimplementing
build+time+parse) means the numbers are directly comparable to
HeCBench-published results.

### `scripts/gen_serial_hecbench.py`
One-shot LLM utility that produces serial CPU versions of HeCBench
benchmarks for use as "serial → parallel" task prompts. For each
`src/<name>-<src-model>/` (default `--src-model omp`), pulls the
main source file (or merges multiple sources when needed), asks the
model to strip parallel directives, and writes the result to
`benchmarks/HeCBench/serial/<name>/main.cpp`. Copies sibling headers
(including cross-directory ones referenced via `-I../xxx/` in the
Makefile) so each serial dir is self-contained. `.meta.json` per
benchmark records the `args` / `regex` / `timeout` / `categories`
from `benchmarks.yaml` so the adapter can build prompts without
re-scanning. Supports `--workers`, `--limit`, `--names`,
`--categories`, `--overwrite`.

### `scripts/gen_hecbench_yaml.py`
Regenerates `benchmarks/HeCBench/benchmarks.yaml` from upstream
`src/<name>-<model>/CMakeLists.txt` + `src/scripts/benchmarks/subset.json`.
Replaces HeCBench's own `tools/generate_metadata.py` whose category
parsing leaks adjacent CMake tokens. `--check` diffs against the
shipped yaml without writing.

### `scripts/view_trace.html`
Single-file HTML viewer for `batch/<task>/trace.json`. Runs entirely in
the browser; no backend.

---

## 11. Invariants and non-goals

### Invariants
1. **No benchmark-specific logic in `loop.py` or `tools/`**. Everything
   benchmark-aware lives in `adapters/` or `scripts/run_<bench>.py`.
2. **Workspace isolation**: each task's filesystem effects are confined
   to its `batch/<task_id>/` directory. Tool dispatch resolves paths
   through `workspace.resolve()` which rejects `..` escapes.
3. **Thread safety via thread-locals**: `workspace.set_root()` and
   `submission.reset()` are thread-local, so `workers: N` in config
   produces N independent agent runs without mutex.
4. **Deterministic re-runs**: with `temperature: 0.0` and `--skip-agent`,
   the eval pipeline produces identical results across invocations. The
   agent stage is only non-deterministic if temperature > 0 or the vLLM
   server is under concurrent load from multiple workers (known vLLM
   quirk).
5. **Crash-safe incremental export**: after each task completes,
   `batch.py` re-serializes the full partial result set. A crash mid-run
   leaves a valid (partial) `agent_output.json`.

### Non-goals
* **No distributed scheduling**: workers are threads in one process. For
  multi-host runs, launch N instances with disjoint `--limit` slices.
* **No sandboxing beyond `subprocess` + `cwd`**: the `bash` tool can
  read anything the process can read. Run in a constrained container
  for untrusted models.
* **No automatic benchmark patching**: known upstream bugs (e.g., ParEval
  07's FFT) are documented in the README's *Known issues / local
  patches* section and must be applied manually when the benchmark
  repo is re-cloned.
* **No caching of model responses**: every run re-issues every prompt.

---

## 12. Extension checklist — adding a benchmark

1. Clone benchmark into `benchmarks/<name>/` (gitignored).
2. Implement `agent/adapters/<name>.py`:
   * `load(path, **kwargs) → list[AgentTask]` — filter + wrap as tasks.
   * `export(results, output_path) → None` — serialize to benchmark's
     native format, including any output normalization.
3. Add `<Name>BenchmarkConfig` dataclass to `agent/config.py` with a
   `from_yaml` classmethod.
4. Register in `agent/batch.py::ADAPTERS`.
5. Write `scripts/run_<name>.py` mirroring `run_pareval.py`: stage 1
   calls `python -m agent.batch --adapter <name> ...`; later stages
   invoke the benchmark's evaluator.
6. Document known bugs / local patches in `DEV_NOTES.md`.
