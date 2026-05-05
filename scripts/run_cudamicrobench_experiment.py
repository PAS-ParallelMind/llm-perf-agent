#!/usr/bin/env python3
"""End-to-end CUDAMicroBench experiment driver."""

from __future__ import annotations

import argparse
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
"""


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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
    args = parser.parse_args()

    problems = args.run_dir / "problems.json"
    output = args.run_dir / "agent_output.json"
    workspace = args.run_dir / "batch"
    run_yaml = args.run_dir / "run.yaml"
    eval_json = args.run_dir / "eval.json"

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


if __name__ == "__main__":
    main()
