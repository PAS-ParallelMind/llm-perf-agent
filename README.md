# ParallelMind Agent

An agentic framework for writing, debugging, and benchmarking parallel code
(CUDA / MPI / OpenMP). Uses OpenAI-compatible tool-calling with any vLLM
backend.

## Project structure

```
agent/
  adapters/
    base.py              # AgentTask, AgentResult, BenchmarkAdapter ABC
    pareval.py           # ParEval adapter (ingest + normalize + export)
  config.py              # YAML config loaders
  engine.py              # OpenAI-compatible client
  loop.py                # tool-calling agent loop → AgentResult
  batch.py               # adapter-driven batch runner
  main.py                # interactive CLI
  prompts.py             # system prompt
  tools/
    base.py              # @tool registry + JSON schema export
    fs.py                # read / write / edit / glob / grep
    bash.py              # sandboxed shell exec
    parallel.py          # nvcc_build / omp_build / mpi_build (compile + run)
    submit.py            # submit_solution (terminates agent loop)
scripts/
  run_pareval.py         # end-to-end: agent → eval → metrics
benchmarks/              # clone benchmark repos here (gitignored)
runs/                    # experiment runs (gitignored)
```

## Configuration

Each run has two YAML files under `runs/<run-name>/`:

**`agent.yaml`** — model + agent settings (same across benchmarks):
```yaml
model:
  name: openai/gpt-oss-120b
  base_url: http://140.112.90.38:8001/v1
  api_key: EMPTY
  temperature: 0.0
  max_tokens: 2048

agent:
  max_steps: 15
  time_budget: 300
  workers: 10
```

**`config.yaml`** — benchmark-specific settings:
```yaml
# ParEval example
problem_set: omp
launch_configs: benchmarks/ParEval/drivers/launch-configs.json
build_timeout: 30
run_timeout: 120
```

## Quick start

```bash
# 1. start a vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --enable-auto-tool-choice --tool-call-parser hermes

# 2. install deps
uv sync  # or: pip install -r requirements.txt

# 3. interactive mode
uv run python -m agent.main \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-Coder-32B-Instruct
```

## Running a benchmark

### ParEval

```bash
# 1. clone benchmark
git clone <pareval-repo> benchmarks/ParEval

# 2. create a run
mkdir -p runs/my_run

cat > runs/my_run/agent.yaml << 'EOF'
model:
  name: openai/gpt-oss-120b
  base_url: http://140.112.90.38:8001/v1
  api_key: EMPTY
  temperature: 0.0
  max_tokens: 2048

agent:
  max_steps: 15
  time_budget: 300
  workers: 10
EOF

cat > runs/my_run/config.yaml << 'EOF'
problem_set: omp
launch_configs: benchmarks/ParEval/drivers/launch-configs.json
build_timeout: 30
run_timeout: 120
EOF

# 3. run (agent → eval → metrics, all in one)
uv run python scripts/run_pareval.py --run-name my_run

# re-run only metrics
uv run python scripts/run_pareval.py --run-name my_run --skip-agent --skip-eval

# debug: first 3 problems only
uv run python scripts/run_pareval.py --run-name my_run --limit 3
```

Output:
```
runs/my_run/
  agent.yaml          # agent config for this run
  config.yaml         # benchmark config for this run
  agent_output.json   # agent output (normalized)
  results.json        # ParEval eval results
  results.csv         # dataframe
  metrics.csv         # final metrics (build@1, pass@1, speedup, etc.)
  batch/              # per-problem trace.json, tool_calls.jsonl, summary.json
  scratch/            # ParEval eval temp files
```

## Adding a new benchmark

1. Clone the benchmark repo into `benchmarks/`
2. Write an adapter in `agent/adapters/<name>.py` implementing `BenchmarkAdapter`
   - `load()`: benchmark prompts → `list[AgentTask]`
   - `export()`: `list[AgentResult]` → benchmark evaluation format
3. Add a benchmark config dataclass in `agent/config.py`
4. Register the adapter in `agent/batch.py` `ADAPTERS` dict
5. Write a run script in `scripts/run_<name>.py`
