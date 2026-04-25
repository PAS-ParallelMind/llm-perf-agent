"""Adapter-driven batch runner.

Usage example (ParEval OMP):
  python -m agent.batch \
    --adapter pareval --adapter-args '{"problem_set": "omp"}' \
    --prompts ParEval/prompts/generation-prompts.json \
    --output runs/my_run/results.json \
    --model openai/gpt-oss-120b \
    --base-url http://140.112.90.38:8001/v1
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console

from . import memory, submission  # noqa: F401  registers memory tools
from .adapters.base import AgentResult, AgentTask, BenchmarkAdapter
from .adapters.hecbench import HeCBenchAdapter
from .adapters.pareval import ParevalAdapter
from .engine import Engine
from .loop import Agent
from .tools import TOOLS  # noqa: F401  registers tools
from .workspace import get_root, set_root

console = Console()

# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    "pareval": ParevalAdapter,
    "hecbench": HeCBenchAdapter,
}

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Adapter-driven agent batch runner")
    # adapter
    ap.add_argument("--adapter", default="pareval",
                    choices=list(ADAPTERS.keys()),
                    help="Benchmark adapter (default pareval)")
    ap.add_argument("--adapter-args", default="{}",
                    help="JSON dict of extra kwargs for adapter.load()")
    # data
    ap.add_argument("--prompts", required=True,
                    help="Path to benchmark prompt file")
    ap.add_argument("--output", required=True,
                    help="Output file (benchmark-native format)")
    # model
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reasoning", action="store_true",
                    help="Model emits a reasoning trace; echo it back each turn")
    ap.add_argument("--system-prompt-file", default=None,
                    help="Path to a text file containing the task-specific "
                         "system prompt (prepended before the memory block)")
    # agent
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--time-budget", type=int, default=300,
                    help="Per-problem wall-clock budget in seconds")
    ap.add_argument("--workspace-root", default="runs/batch",
                    help="Base dir; each problem gets a subdir")
    # run control
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N problems")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip problems already present in --output")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of concurrent workers")
    args = ap.parse_args()

    # --- Load tasks via adapter ---
    adapter_cls = ADAPTERS[args.adapter]
    adapter = adapter_cls()
    adapter_kwargs: dict[str, Any] = json.loads(args.adapter_args)
    if args.limit:
        adapter_kwargs["limit"] = args.limit
    tasks = adapter.load(args.prompts, **adapter_kwargs)
    console.print(f"[bold]{len(tasks)}[/] tasks loaded via {args.adapter} adapter")

    # --- Workspace ---
    ws_base = Path(args.workspace_root).resolve()
    ws_base.mkdir(parents=True, exist_ok=True)

    # --- Skip existing ---
    done_ids: set[str] = set()
    results: list[AgentResult] = []
    out_path = Path(args.output)
    if args.skip_existing and out_path.exists():
        # Peek at exported JSON to find already-done task ids
        try:
            existing = json.loads(out_path.read_text())
            done_ids = {e["name"] for e in existing if e.get("agent_submitted")}
        except Exception:
            pass

    todo: list[tuple[int, AgentTask]] = []
    for i, task in enumerate(tasks):
        if task.id in done_ids:
            console.print(f"[dim][{i+1}/{len(tasks)}] skip {task.id}[/]")
        else:
            todo.append((i, task))

    # --- Engine ---
    engine = Engine(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
    )

    # --- Task-specific system prompt ---
    system_prompt: str | None = None
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text()

    # --- Run ---
    results_lock = threading.Lock()

    def _run_task(idx: int, task: AgentTask) -> AgentResult:
        root = ws_base / task.id
        root.mkdir(parents=True, exist_ok=True)
        # Optionally pre-seed the workspace with files the task declares
        # it needs (e.g. HeCBench reference.h). Copy is shallow — hidden
        # files (.meta.json etc.) are skipped.
        seed_dir = task.metadata.get("seed_dir")
        if seed_dir:
            for p in Path(seed_dir).iterdir():
                if p.is_file() and not p.name.startswith("."):
                    shutil.copy2(p, root / p.name)
        set_root(root)
        console.print(f"[bold][{idx+1}/{len(tasks)}] start {task.id}[/]")
        result = run_one(engine, task,
                         max_steps=args.max_steps,
                         time_budget=args.time_budget,
                         system_prompt=system_prompt)
        console.print(
            f"[{idx+1}/{len(tasks)}] {task.id} "
            f"submitted={result.submitted} "
            f"steps={result.steps} "
            f"elapsed={result.elapsed_s}s"
        )
        with results_lock:
            results.append(result)
            # Incremental export
            adapter.export(results, str(out_path))
        return result

    if args.workers <= 1:
        for idx, task in todo:
            _run_task(idx, task)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_task, idx, task): task.id
                       for idx, task in todo}
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    console.print(f"[red]{tid} failed: {e}[/]")

    # Final export
    adapter.export(results, str(out_path))
    console.print(f"[green]wrote {len(results)} entries to {out_path}[/]")


if __name__ == "__main__":
    main()
