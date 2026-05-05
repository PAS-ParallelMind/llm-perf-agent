#!/usr/bin/env python3
"""Evaluate CUDAMicroBench agent workspaces.

The agent edits files in each per-task workspace. This script rebuilds those
workspaces, optionally runs each benchmark's test command, extracts reported
``time: ...ms`` values, and compares against a baseline materialized from the
original ``problems.json`` seed files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from typing import Any


TIME_RE = re.compile(r"time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
CHECKSUM_RE = re.compile(r"checksum:\s*([^\s,]+)", re.IGNORECASE)


def load_problems(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("problems.json must contain a list")
    return data


def materialize_seed_files(root: Path, seed_files: dict[str, str]) -> None:
    for relpath, content in seed_files.items():
        dst = root / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)


def cuda_env() -> dict[str, str]:
    env = os.environ.copy()
    for prefix in ("/usr/local/cuda", "/opt/cuda"):
        nvcc = Path(prefix) / "bin" / "nvcc"
        if nvcc.exists():
            env["CUDA_HOME"] = prefix
            env["PATH"] = str(nvcc.parent) + os.pathsep + env.get("PATH", "")
            lib64 = Path(prefix) / "lib64"
            if lib64.exists():
                env["LD_LIBRARY_PATH"] = str(lib64) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            break
    return env


def run_command(command: str, cwd: Path, timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=cuda_env(),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return {
            "command": command,
            "exit_code": proc.returncode,
            "output": out[-20000:],
        }
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or "") + (e.stderr or "")) if isinstance(e.stdout, str) else ""
        return {
            "command": command,
            "exit_code": None,
            "timeout": True,
            "output": out[-20000:],
        }


def summarize_output(output: str) -> dict[str, Any]:
    times = [float(x) for x in TIME_RE.findall(output)]
    checksums = CHECKSUM_RE.findall(output)
    return {
        "times_ms": times,
        "median_time_ms": median(times) if times else None,
        "checksums": checksums,
    }


def evaluate_tree(root: Path, metadata: dict[str, Any], timeout: int, run_tests: bool) -> dict[str, Any]:
    build_cmd = metadata["build_command"]
    test_cmd = metadata.get("test_command")

    build = run_command(build_cmd, root, timeout)
    result: dict[str, Any] = {
        "build": build,
        "build_ok": build.get("exit_code") == 0,
    }
    combined = build.get("output", "")

    if run_tests and result["build_ok"] and test_cmd:
        test = run_command(test_cmd, root, timeout)
        result["test"] = test
        result["test_ok"] = test.get("exit_code") == 0
        combined += "\n" + test.get("output", "")
    else:
        result["test"] = None
        result["test_ok"] = None

    result.update(summarize_output(combined))
    return result


def speedup(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> float | None:
    if not baseline:
        return None
    cand = candidate.get("median_time_ms")
    base = baseline.get("median_time_ms")
    if not cand or not base:
        return None
    return round(base / cand, 4)


def evaluate_problem(
    problem: dict[str, Any],
    workspace_root: Path,
    timeout: int,
    run_tests: bool,
    baseline: bool,
) -> dict[str, Any]:
    task_id = str(problem["id"])
    metadata = dict(problem.get("metadata") or {})
    candidate_root = workspace_root / task_id

    entry: dict[str, Any] = {
        "id": task_id,
        "source_dir": metadata.get("source_dir"),
        "workspace": str(candidate_root),
    }
    if not candidate_root.exists():
        entry["error"] = "candidate workspace does not exist"
        return entry

    entry["candidate"] = evaluate_tree(candidate_root, metadata, timeout, run_tests)

    baseline_result = None
    if baseline:
        with tempfile.TemporaryDirectory(prefix=f"cudamicrobench-{task_id}-") as tmp:
            baseline_root = Path(tmp)
            materialize_seed_files(baseline_root, dict(problem.get("seed_files") or {}))
            baseline_result = evaluate_tree(baseline_root, metadata, timeout, run_tests)
        entry["baseline"] = baseline_result

    entry["speedup_vs_baseline"] = speedup(entry["candidate"], baseline_result)
    entry["success"] = bool(
        entry["candidate"].get("build_ok")
        and (entry["candidate"].get("test_ok") in (True, None))
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-tests", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    problems = load_problems(args.problems)
    results = [
        evaluate_problem(
            problem,
            args.workspace_root,
            timeout=args.timeout,
            run_tests=not args.no_tests,
            baseline=not args.no_baseline,
        )
        for problem in problems
    ]
    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r.get("success")),
        "build_ok": sum(1 for r in results if r.get("candidate", {}).get("build_ok")),
        "speedups": {
            r["id"]: r["speedup_vs_baseline"]
            for r in results
            if r.get("speedup_vs_baseline") is not None
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
