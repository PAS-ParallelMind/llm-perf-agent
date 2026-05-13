#!/usr/bin/env python3
"""End-to-end CUDAMicroBench experiment driver."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from build_cudamicrobench_problems import main as build_problems_main


SYSTEM_PROMPT = """You are optimizing CUDA microbenchmarks in a seeded workspace.
Follow the user task's Required workflow exactly. Use the exact build, profile,
and test commands provided there. Do not invent executable names, do not rename
files, and do not compile source filenames that are not present in the
workspace. Use read_file/glob before editing, edit only listed source files,
then submit_solution with a concise summary when build/test/profiling is done.
Measure before and after any optimization with cuda_profile, which uses local
Nsight Compute and can automatically add Nsight Systems CUDA timeline facts when
the ncu profile indicates they are useful. If the candidate is not faster than
the baseline, either revert the change or clearly submit that no improvement was
found; do not claim a speedup without measured evidence.
"""


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_summary(
    *,
    problems: Path,
    agent_output: Path,
    eval_json: Path,
    summary_path: Path,
) -> None:
    problem_rows = load_json(problems, [])
    agent_rows = load_json(agent_output, [])
    eval_doc = load_json(eval_json, {})

    agent_by_id = {row.get("id"): row for row in agent_rows}
    eval_by_id = {
        row.get("id"): row
        for row in eval_doc.get("results", [])
        if isinstance(row, dict)
    }

    tasks = []
    for problem in problem_rows:
        task_id = problem.get("id")
        agent = agent_by_id.get(task_id, {})
        evaluated = eval_by_id.get(task_id, {})
        candidate = evaluated.get("candidate", {}) if isinstance(evaluated, dict) else {}
        baseline = evaluated.get("baseline", {}) if isinstance(evaluated, dict) else {}
        speedup = evaluated.get("speedup_vs_baseline") if isinstance(evaluated, dict) else None
        tasks.append({
            "id": task_id,
            "agent_elapsed_s": agent.get("elapsed_s"),
            "agent_steps": agent.get("steps"),
            "submitted": agent.get("submitted", False),
            "agent_error": agent.get("error"),
            "build_ok": candidate.get("build_ok"),
            "test_ok": candidate.get("test_ok"),
            "timing_valid": candidate.get("timing_valid"),
            "checksum_match": evaluated.get("checksum_match") if isinstance(evaluated, dict) else None,
            "checksum_diagnostics": evaluated.get("checksum_diagnostics") if isinstance(evaluated, dict) else None,
            "success": evaluated.get("success") if isinstance(evaluated, dict) else None,
            "baseline_median_time_ms": baseline.get("median_time_ms"),
            "candidate_median_time_ms": candidate.get("median_time_ms"),
            "speedup_vs_baseline": speedup,
            "speedup_percent": round((speedup - 1.0) * 100, 2) if speedup is not None else None,
            "workspace": evaluated.get("workspace") if isinstance(evaluated, dict) else None,
        })

    comparable = [
        t for t in tasks
        if t["speedup_vs_baseline"] is not None and t["success"]
    ]
    invalid_speedups = [
        t for t in tasks
        if t["speedup_vs_baseline"] is not None and not t["success"]
    ]
    summary = {
        "total_tasks": len(tasks),
        "submitted": sum(1 for t in tasks if t["submitted"]),
        "build_ok": sum(1 for t in tasks if t["build_ok"]),
        "test_ok": sum(1 for t in tasks if t["test_ok"]),
        "timing_valid": sum(1 for t in tasks if t["timing_valid"] is True),
        "timing_invalid": sum(1 for t in tasks if t["timing_valid"] is False),
        "checksum_match": sum(1 for t in tasks if t["checksum_match"] is True),
        "checksum_mismatch": sum(1 for t in tasks if t["checksum_match"] is False),
        "success": sum(1 for t in tasks if t["success"]),
        "comparable_speedups": len(comparable),
        "invalid_speedups": len(invalid_speedups),
        "faster": sum(1 for t in comparable if t["speedup_vs_baseline"] > 1.0),
        "slower": sum(1 for t in comparable if t["speedup_vs_baseline"] < 1.0),
        "unchanged": sum(1 for t in comparable if t["speedup_vs_baseline"] == 1.0),
        "tasks": tasks,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    print(json.dumps({
        "total_tasks": summary["total_tasks"],
        "submitted": summary["submitted"],
        "success": summary["success"],
        "timing_invalid": summary["timing_invalid"],
        "checksum_mismatch": summary["checksum_mismatch"],
        "faster": summary["faster"],
        "slower": summary["slower"],
        "comparable_speedups": summary["comparable_speedups"],
        "invalid_speedups": summary["invalid_speedups"],
    }, indent=2))


def write_run_yaml(args: argparse.Namespace, problems: Path, output: Path, workspace: Path, path: Path) -> None:
    data = {
        "model": {
            "name": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reasoning": args.reasoning,
        },
        "agent": {
            "max_steps": args.max_steps,
            "time_budget": args.time_budget,
            "workers": args.workers,
        },
        "io": {
            "input": str(problems.resolve()),
            "output": str(output.resolve()),
            "workspace_root": str(workspace.resolve()),
        },
        "system_prompt": SYSTEM_PROMPT,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks/CUDAMicroBench"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/cudamicrobench"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--time-budget", type=int, default=900)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        help="Only run these CUDAMicroBench ids, e.g. HDOverlap CoMem_AXPY",
    )
    parser.add_argument(
        "--default-selection",
        action="store_true",
        help="Run the recommended starter set including HDOverlap",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-agent", action="store_true", help="only generate problems and run.yaml")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-no-tests", action="store_true")
    parser.add_argument("--summary-output", type=Path, help="Path for compact run summary JSON")
    args = parser.parse_args()

    problems = args.run_dir / "problems.json"
    output = args.run_dir / "agent_output.json"
    workspace = args.run_dir / "batch"
    run_yaml = args.run_dir / "run.yaml"
    eval_json = args.run_dir / "eval.json"
    summary_json = args.summary_output or (args.run_dir / "summary.json")

    sys.argv = [
        "build_cudamicrobench_problems.py",
        "--benchmark-root",
        str(args.benchmark_root),
        "--output",
        str(problems),
    ]
    if args.default_selection:
        sys.argv.append("--default-selection")
    elif args.include:
        sys.argv.append("--include")
        sys.argv.extend(args.include)
    build_problems_main()
    write_run_yaml(args, problems, output, workspace, run_yaml)
    print(f"wrote {problems}")
    print(f"wrote {run_yaml}")

    if args.no_agent:
        return

    batch_cmd = [sys.executable, "-m", "agent.batch", "--config", str(run_yaml)]
    if args.limit:
        batch_cmd.extend(["--limit", str(args.limit)])
    if args.skip_existing:
        batch_cmd.append("--skip-existing")
    run(batch_cmd)

    if not args.no_eval:
        eval_cmd = [
            sys.executable,
            "scripts/eval_cudamicrobench.py",
            "--problems",
            str(problems),
            "--workspace-root",
            str(workspace),
            "--output",
            str(eval_json),
        ]
        if args.eval_no_tests:
            eval_cmd.append("--no-tests")
        run(eval_cmd)
        write_summary(
            problems=problems,
            agent_output=output,
            eval_json=eval_json,
            summary_path=summary_json,
        )


if __name__ == "__main__":
    main()
