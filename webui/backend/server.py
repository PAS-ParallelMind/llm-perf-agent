"""FastAPI backend for the llm-perf-agent web UI.

Serves the chat-session trace viewer. Each session writes its artefacts
under ``RUNS_DIR/<session-name>/batch/session/``:

  * ``trace.json``       full message history (OpenAI-format)
  * ``tool_calls.jsonl`` step-indexed dispatch log
  * ``summary.json``     name / started_at / agent_model / turns / steps / elapsed

Run locally:
    uv run python -m webui.backend.server
or:
    uvicorn webui.backend.server:app --host 0.0.0.0 --port 8080 --reload

The runs directory defaults to ``<repo>/runs/chat`` (where ``agent.main``
writes chat sessions). Override with ``RUNS_DIR=/path/to/dir``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .chat import SessionStore, cfg_from_form, discover_presets


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HARNESS_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(os.environ.get("RUNS_DIR", HARNESS_ROOT / "runs" / "chat")).resolve()
FRONTEND_DIST = HARNESS_ROOT / "webui" / "frontend" / "dist"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Each chat session uses a fixed single-workspace name under ``batch/``
# (see agent.main._init_session). We don't expose the per-task fanout the
# old kernel-eval harness needed.
_SESSION_WORKSPACE = "session"

# Live chat-session registry shared across requests.
STORE = SessionStore(RUNS_DIR)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="llm-perf-agent webui", version="0.2")

# Permissive CORS so the Vite dev server (port 5173) can hit us during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"not found: {path.name}")
    return json.loads(path.read_text())


def _safe_run_dir(name: str) -> Path:
    """Resolve ``name`` to an absolute path under RUNS_DIR, refusing traversal."""
    if not name or "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail=f"invalid run name: {name!r}")
    d = (RUNS_DIR / name).resolve()
    if not d.is_dir() or RUNS_DIR not in d.parents:
        raise HTTPException(status_code=404, detail=f"run not found: {name}")
    return d


def _session_workspace(run_dir: Path) -> Path:
    """Return ``<run>/batch/session`` if it exists, else 404."""
    ws = run_dir / "batch" / _SESSION_WORKSPACE
    if not ws.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"no session workspace in {run_dir.name}",
        )
    return ws


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """Aggregate top-level info for one chat-session directory.

    Reads ``batch/session/summary.json`` if present, falling back to
    ``run.yaml`` for model/agent metadata. Best-effort: any read error
    leaves the field unset rather than failing the whole listing.
    """
    info: dict[str, Any] = {"name": run_dir.name}

    # modified_at — most recent of the run's core artefacts.
    candidates = [
        run_dir / "batch" / _SESSION_WORKSPACE / "summary.json",
        run_dir / "batch" / _SESSION_WORKSPACE / "trace.json",
        run_dir / "run.yaml",
        run_dir,
    ]
    mtime = max((p.stat().st_mtime for p in candidates if p.exists()),
                default=run_dir.stat().st_mtime)
    info["modified_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    # summary.json — turns / steps / elapsed / agent_model.
    summary_path = run_dir / "batch" / _SESSION_WORKSPACE / "summary.json"
    if summary_path.is_file():
        try:
            s = json.loads(summary_path.read_text())
            info.update({
                "started_at":      s.get("started_at"),
                "agent_model":     s.get("agent_model"),
                "turns":           s.get("turns"),
                "total_steps":     s.get("total_steps"),
                "total_elapsed_s": s.get("total_elapsed_s"),
            })
        except (OSError, json.JSONDecodeError):
            pass

    # run.yaml — for sessions that haven't run a turn yet (no summary.json).
    if "agent_model" not in info:
        yaml_path = run_dir / "run.yaml"
        if yaml_path.is_file():
            try:
                cfg = yaml.safe_load(yaml_path.read_text()) or {}
                info["agent_model"] = (cfg.get("model") or {}).get("name")
            except (OSError, yaml.YAMLError):
                pass

    return info


# ---------------------------------------------------------------------------
# Routes — /api/*
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status":       "ok",
        "harness_root": str(HARNESS_ROOT),
        "runs_dir":     str(RUNS_DIR),
    }


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    """List every chat session under RUNS_DIR, newest first."""
    if not RUNS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in RUNS_DIR.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        # Only count it as a session if it has either a run.yaml or a
        # session workspace.
        if not (child / "run.yaml").is_file() and not (child / "batch" / _SESSION_WORKSPACE).is_dir():
            continue
        out.append(_run_summary(child))
    out.sort(key=lambda r: r.get("modified_at", ""), reverse=True)
    return out


@app.get("/api/runs/{name}")
def run_detail(name: str) -> dict[str, Any]:
    return _run_summary(_safe_run_dir(name))


@app.get("/api/runs/{name}/trace")
def run_trace(name: str) -> Any:
    return _load_json(_session_workspace(_safe_run_dir(name)) / "trace.json")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@app.get("/api/runs/{name}/tool_calls")
def run_tool_calls(name: str) -> list[dict[str, Any]]:
    """Parse the session's ``tool_calls.jsonl`` log."""
    p = _session_workspace(_safe_run_dir(name)) / "tool_calls.jsonl"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="no tool_calls log")
    return _read_jsonl(p)


@app.get("/api/runs/{name}/llm_requests")
def run_llm_requests(name: str) -> list[dict[str, Any]]:
    """Parse the session's ``llm_requests.jsonl`` log — the exact request
    payload (messages + tool schemas + sampling params) sent to the model
    on each LLM call."""
    p = _session_workspace(_safe_run_dir(name)) / "llm_requests.jsonl"
    if not p.is_file():
        # Older sessions predate this artefact — return empty rather than 404
        # so the UI degrades gracefully.
        return []
    return _read_jsonl(p)


# ---------------------------------------------------------------------------
# Live chat — POST endpoints that actually run the agent
# ---------------------------------------------------------------------------

@app.get("/api/presets")
def list_presets() -> list[dict[str, Any]]:
    """Snapshots of every chat-config YAML in the harness root the new-
    session form can pre-fill from. Includes a built-in fake-dry-run
    preset for testing without a vLLM server."""
    return discover_presets(HARNESS_ROOT)


@app.post("/api/sessions")
def create_session(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Create a new chat session and return its summary.

    Body (all fields under model are passed straight to ModelConfig):

        {
          "name":       "optional-session-slug",
          "max_steps":  20,
          "system_prompt":      "...",   // inline, takes precedence
          "system_prompt_file": "prompts/perf_engineer.md",
          "model": {
            "name":              "/path/or/openai-id",
            "base_url":          "http://...",
            "api_key":           "EMPTY",
            "temperature":       0.0,
            "max_output_tokens": 4096,
            "max_model_len":     32768,
            "reasoning":         false
          }
        }
    """
    try:
        cfg = cfg_from_form(payload, RUNS_DIR)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}") from e
    try:
        name = STORE.create(cfg)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _run_summary(_safe_run_dir(name))


@app.post("/api/runs/{name}/chat")
def chat(name: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Send one user message and block until the agent's reply is ready.

    Returns the final assistant text, the steps the loop consumed, the
    elapsed wall time, and the full updated message list so the UI can
    re-render the conversation without a second round-trip."""
    msg = (payload.get("message") or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")
    # Validate the session exists / can be resumed before delegating, so
    # we return a clean 404 instead of a 500 from inside the store.
    _safe_run_dir(name)
    try:
        return STORE.chat(name, msg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        # Engine errors (vLLM down, bad model name, etc.) surface here.
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e


@app.post("/api/runs/{name}/reset")
def reset(name: str) -> dict[str, Any]:
    """Drop conversation history but keep the session dir + engine."""
    _safe_run_dir(name)
    try:
        STORE.reset(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _run_summary(_safe_run_dir(name))


# ---------------------------------------------------------------------------
# Static frontend (only mounted when built)
# ---------------------------------------------------------------------------

if FRONTEND_DIST.is_dir():
    app.mount("/assets",
              StaticFiles(directory=str(FRONTEND_DIST / "assets")),
              name="assets")

    # SPA fallback: any non-/api path that doesn't map to a real file in
    # dist/ returns index.html so React Router can handle the route on a
    # hard refresh.
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        candidate = (FRONTEND_DIST / full_path).resolve()
        if (FRONTEND_DIST in candidate.parents and candidate.is_file()):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("webui.backend.server:app", host="0.0.0.0",
                port=port, reload=True)


if __name__ == "__main__":
    main()
