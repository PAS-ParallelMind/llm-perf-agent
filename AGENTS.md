# Repository Guidelines

## Project Structure & Module Organization

This is a Python harness for agentic parallel-code generation and benchmarking. Core source lives in `agent/`: `batch.py` runs config-driven batches, `main.py` is the experimental interactive CLI, `loop.py` owns the tool-calling loop, and `engine.py` wraps OpenAI-compatible model clients. Tool implementations are under `agent/tools/`. `scripts/` holds benchmark or baseline utilities such as `scripts/run_bare.py`. `visualize_tool/view_trace.html` is a standalone trace viewer. Static README assets live in `assets/`. `SPEC.md` documents contracts and invariants; keep behavior-changing edits consistent with it.

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python -m agent.batch --config /path/to/run.yaml`: run a batch using a YAML config.
- `uv run python -m agent.batch --config /path/to/run.yaml --limit 1`: smoke-test a single problem.
- `uv run python -m agent.main --dry-run`: exercise the interactive loop with `FakeEngine`, without a vLLM server.
- `uv run python scripts/run_bare.py ...`: run the bare-model baseline.
- `python -m http.server` from the repo root: serve `visualize_tool/view_trace.html` when browser file loading is restricted.

## Coding Style & Naming Conventions

Use Python 3.10+ and standard 4-space indentation. Follow existing module style: dataclasses or typed dictionaries for contracts, explicit path handling, and narrow tool modules. Prefer `snake_case` for functions, variables, and modules; use `PascalCase` for classes such as `RunConfig` and `AgentResult`. Keep user-facing JSON/YAML field names stable and documented in `README.md` or `SPEC.md`.

## Testing Guidelines

No dedicated test suite is currently checked in. For changes to the agent loop or tools, run `uv run python -m agent.main --dry-run`. For batch behavior, run a minimal config with `--limit 1` and inspect `trace.json`, `tool_calls.jsonl`, and `summary.json`. Add tests under a future `tests/` directory using `test_*.py` naming for logic that does not need a model server.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, for example `Add bare-model baseline runner` and `Replace adapter layer with JSON-in/JSON-out contract`. Keep the first line focused and under roughly 72 characters. Pull requests should describe the behavior change, list commands or smoke tests run, call out config or schema changes, and include screenshots only when touching `assets/` or `visualize_tool/`.

## Security & Configuration Tips

Do not commit run outputs, model API keys, or machine-specific benchmark data. Prefer `api_key: EMPTY` only for local OpenAI-compatible servers. Keep generated workspaces under configured run directories such as `runs/` or another ignored path.
