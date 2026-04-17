#!/usr/bin/env python3
"""End-to-end ParEval benchmark script.

Reads two configs from runs/<run-name>/:
  - agent.yaml   — model + agent settings
  - config.yaml  — benchmark-specific settings

Usage:
  uv run python3 scripts/run_pareval.py --run-name test
  uv run python3 scripts/run_pareval.py --run-name test --limit 3
  uv run python3 scripts/run_pareval.py --run-name test --skip-agent --skip-eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.config import AgentConfig, ParevalBenchmarkConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAREVAL_ROOT = PROJECT_ROOT / "benchmarks" / "ParEval"
PROMPTS_JSON = PAREVAL_ROOT / "prompts" / "generation-prompts.json"
DRIVERS_DIR = PAREVAL_ROOT / "drivers"
ANALYSIS_DIR = PAREVAL_ROOT / "analysis"
PROBLEM_SIZES = DRIVERS_DIR / "problem-sizes.json"
RUNS_DIR = PROJECT_ROOT / "runs"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nERROR: exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def patch_for_dataframe(results_path: Path) -> None:
    data = json.loads(results_path.read_text())
    for entry in data:
        entry.setdefault("temperature", 0.0)
        entry.setdefault("top_p", 1.0)
        entry.setdefault("do_sample", False)
        entry.setdefault("max_new_tokens", 4096)
    results_path.write_text(json.dumps(data, indent=2))


def patch_csv_num_procs(csv_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "num_procs" not in df.columns:
        df["num_procs"] = 0
        df.to_csv(csv_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end ParEval benchmark: agent → eval → metrics."
    )
    ap.add_argument("--run-name", required=True, help="Run directory name under runs/")
    ap.add_argument("--limit", type=int, default=None, help="Only run first N problems")
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-metrics", action="store_true")
    args = ap.parse_args()

    # --- Load configs ---
    run_dir = RUNS_DIR / args.run_name
    agent_yaml = run_dir / "agent.yaml"
    bench_yaml = run_dir / "config.yaml"
    if not agent_yaml.exists():
        sys.exit(f"ERROR: {agent_yaml} not found. Create it first.")
    if not bench_yaml.exists():
        sys.exit(f"ERROR: {bench_yaml} not found. Create it first.")

    acfg = AgentConfig.from_yaml(agent_yaml)
    bcfg = ParevalBenchmarkConfig.from_yaml(bench_yaml)

    # --- Directories ---
    batch_dir = run_dir / "batch"
    batch_dir.mkdir(exist_ok=True)
    scratch_dir = run_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    agent_output = run_dir / "agent_output.json"
    results_json = run_dir / "results.json"
    results_csv = run_dir / "results.csv"
    metrics_csv = run_dir / "metrics.csv"

    print(f"Run directory: {run_dir}")
    print(f"Model:         {acfg.model.name}")
    print(f"Problem set:   {bcfg.problem_set}")

    # --- Stage 1: Agent ---
    if not args.skip_agent:
        print("\n[1/3] Running agent...")
        adapter_args = json.dumps({"parallelism": bcfg.problem_set})
        cmd = [
            sys.executable, "-m", "agent.batch",
            "--adapter", "pareval",
            "--adapter-args", adapter_args,
            "--prompts", str(PROMPTS_JSON),
            "--output", str(agent_output),
            "--model", acfg.model.name,
            "--base-url", acfg.model.base_url,
            "--api-key", acfg.model.api_key,
            "--temperature", str(acfg.model.temperature),
            "--max-tokens", str(acfg.model.max_tokens),
            *(["--reasoning"] if acfg.model.reasoning else []),
            "--max-steps", str(acfg.agent.max_steps),
            "--time-budget", str(acfg.agent.time_budget),
            "--workspace-root", str(batch_dir),
            "--workers", str(acfg.agent.workers),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run_cmd(cmd, cwd=PROJECT_ROOT)
    else:
        print("\n[1/3] Skipped (--skip-agent)")
        if not agent_output.exists():
            sys.exit(f"ERROR: {agent_output} not found")

    # --- Stage 2: Eval ---
    launch_configs = str(
        PROJECT_ROOT / bcfg.launch_configs
    ) if bcfg.launch_configs else str(DRIVERS_DIR / "launch-configs.json")

    if not args.skip_eval:
        print("\n[2/3] Running ParEval eval...")
        cmd = [
            sys.executable, str(DRIVERS_DIR / "run-all.py"),
            str(agent_output),
            "-o", str(results_json),
            "--scratch-dir", str(scratch_dir),
            "--launch-configs", launch_configs,
            "--include-models", bcfg.problem_set,
            "--build-timeout", str(bcfg.build_timeout),
            "--run-timeout", str(bcfg.run_timeout),
            "--yes-to-all",
            "--overwrite",
        ]
        run_cmd(cmd, cwd=DRIVERS_DIR)
    else:
        print("\n[2/3] Skipped (--skip-eval)")
        if not results_json.exists():
            sys.exit(f"ERROR: {results_json} not found")

    # --- Stage 3: Metrics ---
    if not args.skip_metrics:
        print("\n[3/3] Computing metrics...")
        patch_for_dataframe(results_json)

        run_cmd([
            sys.executable, str(ANALYSIS_DIR / "create-dataframe.py"),
            str(results_json), "-o", str(results_csv),
        ])
        patch_csv_num_procs(results_csv)

        run_cmd([
            sys.executable, str(ANALYSIS_DIR / "metrics.py"),
            str(results_csv), "-k", "1",
            "--problem-sizes", str(PROBLEM_SIZES),
            "-o", str(metrics_csv),
        ])
    else:
        print("\n[3/3] Skipped (--skip-metrics)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Done! → {run_dir}")
    print(f"{'='*60}")
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        cols = [c for c in df.columns if c not in ("model",)]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
