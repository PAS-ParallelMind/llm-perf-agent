#!/usr/bin/env python3
"""Agent-driven HeCBench serial-code generator.

Same goal as ``scripts/gen_serial_hecbench.py`` (produce a serial CPU
reference for each HeCBench benchmark), but framed as an *agentic* task:
the agent gets every available parallel variant (cuda/omp/hip/sycl) of
a benchmark mounted as a subdirectory of its workspace, and iterates via
tool calls (``read_file`` / ``write_file`` / ``cpp_build_and_run`` / ...)
until the serial code compiles + runs, then submits.

Configuration (mirrors ``scripts/run_hecbench.py``)
----------------------------------------------------
Reads two YAML files from ``runs/hecbench_serial_gen/<run-name>/``:

  - ``agent.yaml``   — model + agent settings (same schema as run_hecbench)
  - ``config.yaml``  — gen-specific settings (HeCBenchSerialGenConfig)

Output layout
-------------
Run artifacts (per-benchmark traces, tool calls, summaries, working
files) live in::

    runs/hecbench_serial_gen/<run-name>/
    ├── agent.yaml
    ├── config.yaml
    ├── agent_log.json         ← per-benchmark status (copied from corpus)
    ├── summary.json           ← run-level totals
    └── batch/<name>/
        ├── trace.json
        ├── tool_calls.jsonl
        ├── summary.json
        ├── main.cpp / main    ← agent's working files
        └── cuda/ omp/ hip/ sycl/

Final corpus (consumed by ``scripts/run_hecbench.py``) is separate;
its location is ``HeCBenchSerialGenConfig.out_root`` (default
``benchmarks/HeCBench/serial_agent/``)::

    benchmarks/HeCBench/serial_agent/
    └── <name>/
        ├── main.cpp
        └── .meta.json

Usage
-----
  uv run python3 scripts/gen_serial_hecbench_agent.py --run-name v1
  uv run python3 scripts/gen_serial_hecbench_agent.py --run-name v1 --limit 5
  uv run python3 scripts/gen_serial_hecbench_agent.py --run-name v1 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import AgentConfig, HeCBenchSerialGenConfig

RUNS_DIR = PROJECT_ROOT / "runs" / "hecbench_serial_gen"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Agent-driven HeCBench serial-code generator")
    ap.add_argument("--run-name", required=True,
                    help="Name of the run dir under runs/hecbench_serial_gen/. "
                         "Must contain agent.yaml and config.yaml.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N benchmarks after filtering.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip benchmarks whose main.cpp is already in "
                         "<out_root>/<name>/.")
    args = ap.parse_args()

    # --- Load configs ---
    run_dir = RUNS_DIR / args.run_name
    agent_yaml = run_dir / "agent.yaml"
    bench_yaml = run_dir / "config.yaml"
    if not agent_yaml.exists():
        sys.exit(f"ERROR: {agent_yaml} not found.")
    if not bench_yaml.exists():
        sys.exit(f"ERROR: {bench_yaml} not found.")

    acfg = AgentConfig.from_yaml(agent_yaml)
    bcfg = HeCBenchSerialGenConfig.from_yaml(bench_yaml)

    # --- Paths ---
    batch_dir = run_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    out_root = (PROJECT_ROOT / bcfg.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    src_root = (PROJECT_ROOT / bcfg.src_root).resolve()
    benchmarks_yaml = (PROJECT_ROOT / bcfg.benchmarks_yaml).resolve()

    # --- Skip-existing prefilter ---
    names_filter: list[str] | None = bcfg.names
    if args.skip_existing:
        existing = {p.name for p in out_root.iterdir()
                    if p.is_dir() and (p / "main.cpp").exists()}
        if existing:
            print(f"Skip-existing: {len(existing)} corpus entries already on disk")
        import yaml as _yaml
        bench = _yaml.safe_load(benchmarks_yaml.read_text()) or {}
        all_names = list(bench)
        if names_filter is None:
            names_filter = [n for n in all_names if n not in existing]
        else:
            names_filter = [n for n in names_filter if n not in existing]
        if not names_filter:
            print("Nothing to do — corpus already complete for the requested set.")
            return

    adapter_args: dict = {
        "src_root": str(src_root),
        "max_steps": acfg.agent.max_steps,
        "require_variant": bcfg.require_variant,
    }
    if names_filter:
        adapter_args["names"] = names_filter
    if bcfg.categories:
        adapter_args["categories"] = bcfg.categories

    # System prompt → temp file (agent.batch reads it from disk).
    sp_args: list[str] = []
    if acfg.system_prompt:
        sp_path = run_dir / "system_prompt.txt"
        sp_path.write_text(acfg.system_prompt)
        sp_args = ["--system-prompt-file", str(sp_path)]

    # --- Build the batch.py invocation ---
    cmd = [
        sys.executable, "-m", "agent.batch",
        "--adapter", "hecbench_serial_gen",
        "--adapter-args", json.dumps(adapter_args),
        "--prompts", str(benchmarks_yaml),
        "--output", str(out_root),
        "--model", acfg.model.name,
        "--base-url", acfg.model.base_url,
        "--api-key", acfg.model.api_key,
        "--temperature", str(acfg.model.temperature),
        "--max-tokens", str(acfg.model.max_tokens),
        "--max-steps", str(acfg.agent.max_steps),
        "--time-budget", str(acfg.agent.time_budget),
        "--workspace-root", str(batch_dir),
        "--workers", str(acfg.agent.workers),
        *sp_args,
    ]
    if acfg.model.reasoning:
        cmd.append("--reasoning")
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    print("\n" + "=" * 70)
    print("  Agent-driven HeCBench serial generation")
    print("=" * 70)
    print(f"  run-name      : {args.run_name}")
    print(f"  run dir       : {run_dir}")
    print(f"  corpus dir    : {out_root}")
    print(f"  src_root      : {src_root}")
    print(f"  yaml          : {benchmarks_yaml}")
    print(f"  model         : {acfg.model.name}")
    print(f"  workers       : {acfg.agent.workers}")
    print(f"  max-steps     : {acfg.agent.max_steps}")
    print(f"  time-budget   : {acfg.agent.time_budget}s")
    if names_filter:
        head = ", ".join(names_filter[:5])
        more = f" + {len(names_filter)-5} more" if len(names_filter) > 5 else ""
        print(f"  names         : {head}{more}")
    if bcfg.categories:
        print(f"  categories    : {bcfg.categories}")
    print("=" * 70 + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")
    rc = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT).returncode
    if rc != 0:
        sys.exit(rc)

    # --- Persist a copy of the adapter's log into the run dir ---
    src_log = out_root / "_agent_log.json"
    log: list[dict] = []
    if src_log.exists():
        log = json.loads(src_log.read_text())
        shutil.copy2(src_log, run_dir / "agent_log.json")

    # Run-level summary
    if log:
        ok = sum(1 for e in log if e.get("status") == "ok")
        no_sub = sum(1 for e in log if e.get("status") == "no-submission")
        empty = sum(1 for e in log if e.get("status") == "empty-code")
        avg_steps = sum(e.get("steps", 0) for e in log) / len(log)
        avg_elapsed = sum(e.get("elapsed_s", 0) for e in log) / len(log)
        summary = {
            "run_name": args.run_name,
            "total": len(log),
            "ok": ok,
            "no_submission": no_sub,
            "empty_code": empty,
            "avg_steps": round(avg_steps, 2),
            "avg_elapsed_s": round(avg_elapsed, 2),
            "corpus_dir": str(out_root),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print("\n" + "=" * 70)
        print("  Summary")
        print("=" * 70)
        print(f"  total       : {len(log)}")
        print(f"  ok          : {ok}")
        print(f"  no-submit   : {no_sub}")
        print(f"  empty-code  : {empty}")
        print(f"  avg steps   : {avg_steps:.1f}")
        print(f"  avg elapsed : {avg_elapsed:.1f}s")
        print(f"  run dir     : {run_dir}")
        print(f"  corpus      : {out_root}")


if __name__ == "__main__":
    main()
