# Repository Guidelines

## Project Structure & Module Organization

This is a Python chat agent for LLM inference deployment guidance and performance analysis. Core source lives in `agent/`: `main.py` is the interactive REPL entry point, `loop.py` owns the multi-turn tool-calling loop (`ChatAgent`), `engine.py` wraps OpenAI-compatible model clients, and `config.py` defines `ChatConfig`. Tool implementations are under `agent/tools/`: `fs.py` / `bash.py` for general I/O, `benchmarking/benchmark.py` (still a stub — endpoint probe), and `modeling/{memory,latency,serving}.py` (real analytical tools: `estimate_memory`, `simulate_serving`; `latency.py` is plumbing used by serving) sharing `modeling/report.py` and `modeling/configs/` for GPU/model presets. Static README assets live in `assets/`. `SPEC.md` documents contracts and invariants; keep behavior-changing edits consistent with it.

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python -m agent.main --dry-run`: exercise the chat loop with `FakeEngine`, no vLLM required.
- `uv run python -m agent.main --config /path/to/chat.yaml`: real session against an OpenAI-compatible endpoint.
- `uv run python -m agent.main --model <name> --base-url <url>`: same, without a YAML config.

## Coding Style & Naming Conventions

Use Python 3.10+ and standard 4-space indentation. Follow existing module style: dataclasses or typed dictionaries for contracts, explicit path handling, and narrow tool modules. Prefer `snake_case` for functions, variables, and modules; use `PascalCase` for classes such as `RunConfig` and `AgentResult`. Keep user-facing JSON/YAML field names stable and documented in `README.md` or `SPEC.md`.

## Testing Guidelines

No dedicated test suite is currently checked in. For changes to the agent loop or tools, run `uv run python -m agent.main --dry-run` and inspect `trace.json`, `tool_calls.jsonl`, and `summary.json` under `runs/`. For modeling-tool changes, the modules under `agent/tools/modeling/` are runnable as standalone CLIs (`uv run python -m agent.tools.modeling.memory ...`) for quick sanity checks. Add tests under a future `tests/` directory using `test_*.py` naming for logic that does not need a model server.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, for example `Refocus repo on LLM inference perf agent` and `Engine: per-call timeout + retry on transient errors`. Keep the first line focused and under roughly 72 characters. Pull requests should describe the behavior change, list commands or smoke tests run, call out config or schema changes, and include screenshots only when touching `assets/` or the `webui/` frontend.

## Security & Configuration Tips

Do not commit run outputs, model API keys, or machine-specific benchmark data. Prefer `api_key: EMPTY` only for local OpenAI-compatible servers. Keep generated workspaces under configured run directories such as `runs/` or another ignored path.
