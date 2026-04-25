#!/usr/bin/env python3
"""End-to-end bare-model ParEval benchmark: generate -> eval -> metrics.

Mirrors scripts/run_pareval.py but replaces stage 1 (agent loop) with a
single-shot call to ParEval's own generate-openai-vllm.py so scoring stays
byte-identical to the upstream baseline. Reads two configs from
runs/<run-name>/:
  - agent.yaml   model settings (model.* used; agent.workers ignored)
  - config.yaml  benchmark-specific settings (same schema as run_pareval.py)

Usage:
  uv run python3 scripts/run_bare.py --run-name bare_omp
  uv run python3 scripts/run_bare.py --run-name bare_omp --limit 3
  uv run python3 scripts/run_bare.py --run-name bare_omp --skip-gen --skip-eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.adapters.pareval import normalize as pareval_normalize
from agent.config import AgentConfig, ParevalBenchmarkConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAREVAL_ROOT = PROJECT_ROOT / "benchmarks" / "ParEval"
PROMPTS_JSON = PAREVAL_ROOT / "prompts" / "generation-prompts.json"
GENERATE_DIR = PAREVAL_ROOT / "generate"
DRIVERS_DIR = PAREVAL_ROOT / "drivers"
ANALYSIS_DIR = PAREVAL_ROOT / "analysis"
PROBLEM_SIZES = DRIVERS_DIR / "problem-sizes.json"
RUNS_DIR = PROJECT_ROOT / "runs" / "pareval"

GENERATE_SCRIPT = GENERATE_DIR / "generate-openai-vllm.py"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nERROR: exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def verify_eval_complete(results_path: Path) -> None:
    """Fail loudly if run-all.py left any outputs as raw strings (un-evaluated)."""
    data = json.loads(results_path.read_text())
    bad = [e["name"] for e in data
           if not all(isinstance(o, dict) for o in e.get("outputs", []))]
    if bad:
        sys.exit(
            f"ERROR: eval incomplete for {len(bad)} problems "
            f"(outputs still raw strings): "
            f"{bad[:5]}{'...' if len(bad) > 5 else ''}"
        )


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


def write_subset_prompts(problem_set: str, limit: int | None, dst: Path) -> int:
    prompts = json.loads(PROMPTS_JSON.read_text())
    subset = [p for p in prompts if p.get("parallelism_model") == problem_set]
    if limit:
        subset = subset[:limit]
    dst.write_text(json.dumps(subset, indent=2))
    return len(subset)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end bare-model ParEval: generate -> eval -> metrics."
    )
    ap.add_argument("--run-name", required=True, help="Run directory name under runs/")
    ap.add_argument("--limit", type=int, default=None, help="Only run first N problems")
    ap.add_argument("--num-samples-per-prompt", type=int, default=1,
                    help="Samples per prompt (needs temperature > 0 for variance)")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("-k", type=int, default=1, help="pass@k for metrics")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-metrics", action="store_true")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_name
    agent_yaml = run_dir / "agent.yaml"
    bench_yaml = run_dir / "config.yaml"
    if not agent_yaml.exists():
        sys.exit(f"ERROR: {agent_yaml} not found. Create it first.")
    if not bench_yaml.exists():
        sys.exit(f"ERROR: {bench_yaml} not found. Create it first.")

    acfg = AgentConfig.from_yaml(agent_yaml)
    bcfg = ParevalBenchmarkConfig.from_yaml(bench_yaml)

    scratch_dir = run_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    prompts_subset = run_dir / "prompts.json"
    agent_output = run_dir / "agent_output.json"
    results_json = run_dir / "results.json"
    results_csv = run_dir / "results.csv"
    metrics_csv = run_dir / "metrics.csv"

    print(f"Run directory: {run_dir}")
    print(f"Model:         {acfg.model.name}")
    print(f"Problem set:   {bcfg.problem_set}")

    # --- Stage 1: Generation (ParEval's own generate-openai-vllm.py) ---
    if not args.skip_gen:
        print("\n[1/3] Generating bare-model outputs...")
        n = write_subset_prompts(bcfg.problem_set, args.limit, prompts_subset)
        print(f"[gen] {n} {bcfg.problem_set} prompts -> {prompts_subset}")
        cmd = [
            sys.executable, str(GENERATE_SCRIPT),
            "-m", acfg.model.name,
            "-p", str(prompts_subset),
            "-o", str(agent_output),
            "--base-url", acfg.model.base_url,
            "--api-key", acfg.model.api_key,
            "--temperature", str(acfg.model.temperature),
            "--top-p", str(args.top_p),
            "--max-new-tokens", str(acfg.model.max_tokens),
            "--num-samples-per-prompt", str(args.num_samples_per_prompt),
            "--overwrite",
        ]
        run_cmd(cmd, cwd=GENERATE_DIR)
        # Post-process: reuse the adapter's normalize so prompt+output compiles
        # (strips markdown fence, re-emitted signature, and outer braces).
        data = json.loads(agent_output.read_text())
        for entry in data:
            prompt = entry.get("prompt", "")
            entry["outputs"] = [pareval_normalize(prompt, o) for o in entry["outputs"]]
        agent_output.write_text(json.dumps(data, indent=2))
    else:
        print("\n[1/3] Skipped (--skip-gen)")
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
        verify_eval_complete(results_json)
        patch_for_dataframe(results_json)

        run_cmd([
            sys.executable, str(ANALYSIS_DIR / "create-dataframe.py"),
            str(results_json), "-o", str(results_csv),
        ])
        patch_csv_num_procs(results_csv)

        run_cmd([
            sys.executable, str(ANALYSIS_DIR / "metrics.py"),
            str(results_csv), "-k", str(args.k),
            "--problem-sizes", str(PROBLEM_SIZES),
            "-o", str(metrics_csv),
        ])
    else:
        print("\n[3/3] Skipped (--skip-metrics)")

    print(f"\n{'='*60}")
    print(f"  Done! -> {run_dir}")
    print(f"{'='*60}")
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        cols = [c for c in df.columns if c not in ("model",)]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
