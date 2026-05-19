# webui — integrated harness frontend

Interactive replacement for the three legacy viewers:

| Page | Replaces |
|---|---|
| **Analyzer** | `visualize_tool/render_comparison.py` (PNG) |
| **Benchmarks** | `eval/view_benchmarks.html` |
| **Trace** | `visualize_tool/view_trace.html` |

## Architecture

- **Backend** (`webui/backend/`): FastAPI + uvicorn. Reads `runs/<name>/...`
  and `eval/benchmarks.json` directly from disk; exposes everything under
  `/api/*`.
- **Frontend** (`webui/frontend/`): Vite + React + TypeScript + Tailwind.
  Built artefacts in `webui/frontend/dist/` are mounted at `/` by the
  backend so the whole UI is one origin in production.

## Endpoints

```
GET  /api/health                              health probe
GET  /api/benchmarks                          full benchmarks.json
GET  /api/runs                                list summaries for every
                                              run dir under runs/
GET  /api/runs/{name}                         summary for one run
GET  /api/runs/{name}/agent_output            agent_output.json
GET  /api/runs/{name}/eval_results            eval_results.json
GET  /api/runs/{name}/batch/{pid}/trace       per-task trace.json
GET  /api/runs/{name}/batch/{pid}/code        submitted main.cu (text)
```

## Dev

Two terminals:

```bash
# 1. backend (with auto-reload)
cd llm-perf-agent
/mnt/disk2/elton7318/venv/bin/python -m webui.backend.server   # :8080

# 2. frontend (Vite dev server, proxies /api → :8080)
cd llm-perf-agent/webui/frontend
npm run dev                                                     # :5173
```

Open <http://localhost:5173>. Edits to `webui/frontend/src/**` hot-reload;
edits under `webui/backend/` trigger uvicorn reload.

## Build for "production"

```bash
cd llm-perf-agent/webui/frontend
npm run build                                                   # → dist/

cd ../..
/mnt/disk2/elton7318/venv/bin/python -m webui.backend.server   # serves
                                                                # both /api
                                                                # and /
```

Single process on a single port. Mount `webui/backend/server.py` behind a
reverse proxy if you need to expose it externally.

## Port

Override via `PORT` env var (default `8080`):

```bash
PORT=9099 /mnt/disk2/elton7318/venv/bin/python -m webui.backend.server
```
