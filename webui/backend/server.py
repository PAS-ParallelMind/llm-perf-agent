"""FastAPI backend for the llm-perf-agent web UI.

Run locally:
    /mnt/disk2/elton7318/venv/bin/python -m webui.backend.server
or:
    uvicorn webui.backend.server:app --host 0.0.0.0 --port 8080 --reload

Endpoints under ``/api``; built frontend (Vite output) will be mounted
at ``/`` once it exists.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HARNESS_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = HARNESS_ROOT / "runs"
BENCH_PATH = HARNESS_ROOT / "eval" / "benchmarks.json"
FRONTEND_DIST = HARNESS_ROOT / "webui" / "frontend" / "dist"

# Run dirs we should ignore when listing (legacy / non-session layouts).
_RUN_BLOCKLIST = {"legacy", "hecbench", "hecbench_serial_gen", "pareval"}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="llm-perf-agent webui", version="0.1")

# Permissive CORS so Vite dev server (port 5173) can hit us during dev.
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
    """Resolve a run name to an absolute path within RUNS_DIR, refusing
    traversal."""
    if name in _RUN_BLOCKLIST or "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail=f"invalid run name: {name!r}")
    d = (RUNS_DIR / name).resolve()
    if not d.is_dir() or RUNS_DIR not in d.parents:
        raise HTTPException(status_code=404, detail=f"run not found: {name}")
    return d


def _all_pass(summary: dict[str, Any]) -> bool:
    if summary["total"] < 1:
        return False
    return (summary["pass_byte"]
            + summary["pass_checker"]
            + summary["pass_llm"]) == summary["total"]


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """Aggregate top-level numbers for one run directory (best-effort)."""
    info: dict[str, Any] = {"name": run_dir.name}

    # modified_at — based on the most-recently-written core artifact
    candidate_files = [
        run_dir / "eval_results.json",
        run_dir / "agent_output.json",
        run_dir / "run.yaml",
        run_dir,
    ]
    mtime = max((f.stat().st_mtime for f in candidate_files if f.exists()),
                default=run_dir.stat().st_mtime)
    info["modified_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    # model + agent metadata from run.yaml (any filename ending in .yaml)
    yaml_path = run_dir / "run.yaml"
    if not yaml_path.is_file():
        # fallback: pick first .yaml in the dir
        yamls = sorted(run_dir.glob("*.yaml"))
        yaml_path = yamls[0] if yamls else None     # type: ignore
    if yaml_path and yaml_path.is_file():
        try:
            cfg = yaml.safe_load(yaml_path.read_text()) or {}
            info["model_name"] = (cfg.get("model") or {}).get("name")
            info["workers"]    = (cfg.get("agent") or {}).get("workers")
        except Exception:
            pass

    # agent_output summary
    ao = run_dir / "agent_output.json"
    if not ao.is_file():
        # bare runs sometimes use a different filename
        for f in run_dir.glob("*_output.json"):
            ao = f
            break
    if ao.is_file():
        try:
            entries = json.loads(ao.read_text())
            info["n_total"] = len(entries)
            info["n_submitted"] = sum(1 for e in entries if e.get("submitted"))
            info["agent_output_path"] = ao.name
        except Exception:
            pass

    # eval_results summary
    er = run_dir / "eval_results.json"
    if er.is_file():
        info["has_eval"] = True
        try:
            results = json.loads(er.read_text())
            info["n_pass"] = sum(
                1 for r in results
                if r.get("validation")
                and _all_pass(r["validation"]["summary"])
            )
        except Exception:
            pass
    else:
        info["has_eval"] = False

    return info


# ---------------------------------------------------------------------------
# Routes — /api/*
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok",
            "harness_root": str(HARNESS_ROOT),
            "runs_dir":     str(RUNS_DIR)}


@app.get("/api/benchmarks")
def benchmarks() -> Any:
    """Return ``eval/benchmarks.json`` verbatim."""
    return _load_json(BENCH_PATH)


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    """List every run dir under ``runs/`` (excluding the legacy/per-suite
    folders) with a top-level summary."""
    if not RUNS_DIR.is_dir():
        return []
    out = []
    for child in sorted(RUNS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if child.name in _RUN_BLOCKLIST:
            continue
        # Skip dirs that have neither agent_output nor run.yaml — probably
        # not a run.
        if not any((child / f).is_file() for f in
                   ("run.yaml", "agent_output.json", "bare_output.json")):
            continue
        out.append(_run_summary(child))
    return out


@app.get("/api/runs/{name}")
def run_detail(name: str) -> dict[str, Any]:
    return _run_summary(_safe_run_dir(name))


@app.get("/api/runs/{name}/agent_output")
def run_agent_output(name: str) -> Any:
    d = _safe_run_dir(name)
    for f in ("agent_output.json", "bare_output.json"):
        p = d / f
        if p.is_file():
            return _load_json(p)
    raise HTTPException(status_code=404, detail="no agent_output in run")


@app.get("/api/runs/{name}/eval_results")
def run_eval_results(name: str) -> Any:
    return _load_json(_safe_run_dir(name) / "eval_results.json")


@app.get("/api/runs/{name}/batch/{pid}/trace")
def run_task_trace(name: str, pid: str) -> Any:
    return _load_json(_safe_run_dir(name) / "batch" / pid / "trace.json")


@app.get("/api/runs/{name}/batch/{pid}/tool_calls")
def run_task_tool_calls(name: str, pid: str) -> list[dict[str, Any]]:
    """Parse the per-task ``tool_calls.jsonl`` log. Each line is one
    tool dispatch with ``step``, ``tool``, ``arguments``, ``result``,
    and ``elapsed_ms``."""
    p = _safe_run_dir(name) / "batch" / pid / "tool_calls.jsonl"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="no tool_calls log")
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@app.get("/api/runs/{name}/batch/{pid}/code", response_class=PlainTextResponse)
def run_task_code(name: str, pid: str) -> str:
    """Return the submitted source file from a per-task workspace, if
    present. Tries ``main.cu`` first, then any other ``main.*``."""
    task = _safe_run_dir(name) / "batch" / pid
    if not task.is_dir():
        raise HTTPException(status_code=404, detail=f"no batch dir for {pid}")
    for candidate in ("main.cu", "main.cpp", "main.c"):
        p = task / candidate
        if p.is_file():
            return p.read_text()
    # otherwise, pick first source file
    for ext in (".cu", ".cpp", ".c"):
        for p in task.glob(f"*{ext}"):
            return p.read_text()
    raise HTTPException(status_code=404, detail="no source file in workspace")


# ---------------------------------------------------------------------------
# Static frontend (only mounted when built)
# ---------------------------------------------------------------------------

if FRONTEND_DIST.is_dir():
    # Serve hashed asset bundles directly.
    app.mount("/assets",
              StaticFiles(directory=str(FRONTEND_DIST / "assets")),
              name="assets")

    # SPA fallback: any non-/api path that doesn't map to a real file in
    # dist/ returns index.html so React Router can handle the route on
    # a hard refresh.
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        candidate = (FRONTEND_DIST / full_path).resolve()
        if (FRONTEND_DIST in candidate.parents
                and candidate.is_file()):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("webui.backend.server:app", host="0.0.0.0",
                port=port, reload=True)


if __name__ == "__main__":
    main()
