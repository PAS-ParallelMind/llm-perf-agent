"""Batch runner — config-driven, JSON in / JSON out.

  python -m agent.batch --config /path/to/run.yaml
    [--limit N] [--skip-existing]

run.yaml schema:    RunConfig in agent/config.py
problems.json:      list[{id, prompt, seed_files?, metadata?}]
output JSON:        list[{id, code, submitted, steps, elapsed_s, error,
                          metadata?}]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console

from . import memory, submission  # noqa: F401  registers memory tools
from .engine import Engine
from .loop import Agent
from .tools import TOOLS  # noqa: F401  registers tools
from .types import AgentResult, AgentTask
from .workspace import get_root, set_root

console = Console()

# ---------------------------------------------------------------------------
# Fallback code extraction (when agent forgets submit_solution)
# ---------------------------------------------------------------------------

_FENCED_RE = re.compile(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL)


def _extract_last_code(text: str) -> str:
    blocks = _FENCED_RE.findall(text)
    return blocks[-1].strip() if blocks else text.strip()


# ---------------------------------------------------------------------------
# Per-problem runner
# ---------------------------------------------------------------------------

def run_one(
    engine: Engine,
    task: AgentTask,
    *,
    max_steps: int,
    time_budget: int,
    system_prompt: str | None = None,
) -> AgentResult:
    submission.reset()
    agent = Agent(engine, max_steps=max_steps, system_prompt=system_prompt)

    try:
        result = agent.run(task, time_budget_s=time_budget)
    except Exception as e:
        result = AgentResult(
            task_id=task.id,
            code="",
            raw_reply="",
            steps=agent.step_count,
            elapsed_s=0.0,
            submitted=False,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            metadata=dict(task.metadata),
        )

    # Fallback: if agent didn't submit, extract last code block from reply
    if not result.code and result.raw_reply:
        result.code = _extract_last_code(result.raw_reply)

    # Save trace to workspace
    _save_logs(result)

    return result


def _save_logs(result: AgentResult) -> None:
    """Save trace, tool call log, and summary to the workspace directory."""
    root = get_root()
    try:
        (root / "trace.json").write_text(
            json.dumps(result.trace, indent=2, default=str)
        )
    except Exception:
        pass
    try:
        with open(root / "tool_calls.jsonl", "w") as f:
            for tc in result.tool_calls:
                f.write(json.dumps(tc, default=str) + "\n")
    except Exception:
        pass
    try:
        summary = {
            "task_id": result.task_id,
            "steps": result.steps,
            "elapsed_s": result.elapsed_s,
            "submitted": result.submitted,
            "code_length": len(result.code),
            "error": result.error,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# New mode — config-driven, JSON in / JSON out, no adapter
# ---------------------------------------------------------------------------

def _load_problems_json(path: str) -> tuple[list[AgentTask], dict[str, dict[str, str]]]:
    """Read problems.json and return (tasks, seed_map).

    seed_map keeps seed_files outside the AgentTask so the result's metadata
    stays clean (passthrough) and we don't risk serializing file contents
    twice in output.json."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top level must be a list of problem objects")

    tasks: list[AgentTask] = []
    seed_map: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            raise ValueError(f"{path}[{i}]: must be an object")
        for req in ("id", "prompt"):
            if req not in p:
                raise ValueError(f"{path}[{i}]: missing required key {req!r}")
        pid = str(p["id"])
        if pid in seen:
            raise ValueError(f"{path}[{i}]: duplicate id {pid!r}")
        seen.add(pid)
        tasks.append(AgentTask(
            id=pid,
            instruction=str(p["prompt"]),
            metadata=dict(p.get("metadata") or {}),
        ))
        seeds = p.get("seed_files") or {}
        if seeds:
            if not isinstance(seeds, dict):
                raise ValueError(f"{path}[{i}].seed_files: must be a dict")
            seed_map[pid] = {str(k): str(v) for k, v in seeds.items()}
    return tasks, seed_map


def _write_seed_files(root: Path, seeds: dict[str, str]) -> None:
    """Materialize problems.json's seed_files into the workspace.

    Keys are forward-slash relative paths; nested directories are created.
    Refuses absolute or `..` paths to keep writes inside the workspace."""
    for relpath, content in seeds.items():
        p = Path(relpath)
        if p.is_absolute() or any(part == ".." for part in p.parts):
            raise ValueError(f"seed_files key must stay inside workspace: {relpath!r}")
        dst = root / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)


def _export_results(results: list[AgentResult], output_path: str) -> None:
    """Unified output schema: [{id, code, submitted, steps, elapsed_s, error,
    metadata?}, ...]. Auxiliary <stem>.code.json holds the pure {id, code}
    list for downstream eval."""
    out = []
    for r in results:
        entry: dict[str, Any] = {
            "id":         r.task_id,
            "code":       r.code,
            "submitted":  r.submitted,
            "steps":      r.steps,
            "elapsed_s":  r.elapsed_s,
            "error":      r.error,
        }
        # Drop harness-internal additions (e.g. workspace path) before
        # emitting; only keep keys the caller put in problems.json metadata.
        meta = {k: v for k, v in r.metadata.items() if k != "workspace"}
        if meta:
            entry["metadata"] = meta
        out.append(entry)
    Path(output_path).write_text(json.dumps(out, indent=2))

    p = Path(output_path)
    code_path = p.parent / (p.stem + ".code.json")
    code_only = [
        {"id": r.task_id, "code": r.code}
        for r in results if r.submitted and r.code
    ]
    code_path.write_text(json.dumps(code_only, indent=2))


def run_from_config(config_path: str,
                     *,
                     limit: int | None = None,
                     skip_existing: bool = False) -> None:
    from .config import RunConfig
    cfg = RunConfig.from_yaml(config_path)

    tasks, seed_map = _load_problems_json(cfg.io.input)
    console.print(f"[bold]{len(tasks)}[/] tasks loaded from {cfg.io.input}")

    ws_base = Path(cfg.io.workspace_root).resolve()
    ws_base.mkdir(parents=True, exist_ok=True)

    out_path = Path(cfg.io.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Skip existing ---
    # Seed `results` with previously-successful entries so re-export
    # preserves them. Otherwise --skip-existing would clobber prior
    # successes the moment any new result is written.
    done_ids: set[str] = set()
    preserved: list[AgentResult] = []
    if skip_existing and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            for e in existing:
                if not e.get("submitted"):
                    continue
                preserved.append(AgentResult(
                    task_id=e["id"],
                    code=e.get("code", ""),
                    raw_reply="",
                    steps=e.get("steps", 0),
                    elapsed_s=e.get("elapsed_s", 0.0),
                    submitted=True,
                    error=e.get("error"),
                    metadata=dict(e.get("metadata") or {}),
                ))
                done_ids.add(e["id"])
        except Exception:
            pass

    todo: list[tuple[int, AgentTask]] = []
    for i, task in enumerate(tasks):
        if task.id in done_ids:
            console.print(f"[dim][{i+1}/{len(tasks)}] skip {task.id}[/]")
        else:
            todo.append((i, task))
        if limit and len(todo) >= limit:
            break

    # --- Engine ---
    engine = Engine(
        model=cfg.model.name,
        base_url=cfg.model.base_url,
        api_key=cfg.model.api_key,
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
        reasoning=cfg.model.reasoning,
    )

    system_prompt: str | None = cfg.system_prompt
    if system_prompt is None and cfg.system_prompt_file:
        system_prompt = Path(cfg.system_prompt_file).read_text()

    # --- Run ---
    results: list[AgentResult] = list(preserved)
    results_lock = threading.Lock()

    def _run_task(idx: int, task: AgentTask) -> AgentResult:
        root = ws_base / task.id
        root.mkdir(parents=True, exist_ok=True)
        _write_seed_files(root, seed_map.get(task.id, {}))
        set_root(root)
        console.print(f"[bold][{idx+1}/{len(tasks)}] start {task.id}[/]")
        result = run_one(engine, task,
                         max_steps=cfg.agent.max_steps,
                         time_budget=cfg.agent.time_budget,
                         system_prompt=system_prompt)
        result.metadata["workspace"] = str(root)
        console.print(
            f"[{idx+1}/{len(tasks)}] {task.id} "
            f"submitted={result.submitted} "
            f"steps={result.steps} "
            f"elapsed={result.elapsed_s}s"
        )
        with results_lock:
            results.append(result)
            _export_results(results, str(out_path))
        return result

    if cfg.agent.workers <= 1:
        for idx, task in todo:
            _run_task(idx, task)
    else:
        with ThreadPoolExecutor(max_workers=cfg.agent.workers) as pool:
            futures = {pool.submit(_run_task, idx, task): task.id
                       for idx, task in todo}
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    console.print(f"[red]{tid} failed: {e}[/]")

    _export_results(results, str(out_path))
    console.print(f"[green]wrote {len(results)} entries to {out_path}[/]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Agent batch runner")
    ap.add_argument("--config", required=True,
                    help="run.yaml — see RunConfig in agent/config.py")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N problems")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip ids already submitted in --output")
    args = ap.parse_args()
    run_from_config(args.config,
                    limit=args.limit,
                    skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
