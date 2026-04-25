#!/usr/bin/env python3
"""End-to-end HeCBench benchmark script.

Reads two configs from runs/<run-name>/:
  - agent.yaml   — model + agent settings (same schema as ParEval runs)
  - config.yaml  — HeCBench-specific settings (see HeCBenchBenchmarkConfig)

Pipeline
--------
Stage 1 (agent):     agent.batch produces agent_output.json — per-benchmark
                     records including the agent's candidate main source.
Stage 2 (scratch):   for each submitted entry, mirror the reference
                     <src_root>/<name>-<target>/ directory into
                     <run>/scratch/<name>-<target>/ and overwrite the main
                     source with the agent's submission.
Stage 3 (timing):    invoke HeCBench's own ``autohecbench.py`` twice — once
                     against the reference tree (baseline.csv) and once
                     against the scratch tree (candidate.csv). Each CSV row
                     has N timing samples (N = config.repeat).
Stage 4 (compare):   call ``autohecbench-compare.py baseline.csv
                     candidate.csv`` to produce speedup.md, and merge
                     everything into results.json / results.csv.

Usage
-----
  uv run python3 scripts/run_hecbench.py --run-name my_run
  uv run python3 scripts/run_hecbench.py --run-name my_run --limit 5
  uv run python3 scripts/run_hecbench.py --run-name my_run --skip-agent
  uv run python3 scripts/run_hecbench.py --run-name my_run --skip-scratch
  uv run python3 scripts/run_hecbench.py --run-name my_run --skip-timing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.config import AgentConfig, HeCBenchBenchmarkConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "hecbench"

AUTOHECBENCH = PROJECT_ROOT / "benchmarks/HeCBench/src/scripts/autohecbench.py"
AUTOHECBENCH_CMP = PROJECT_ROOT / "benchmarks/HeCBench/src/scripts/autohecbench-compare.py"


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"\nERROR: exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Stage 2: mirror reference tree + overwrite agent source
# ---------------------------------------------------------------------------

def prepare_scratch_tree(
    entries: list[dict[str, Any]],
    *,
    src_root: Path,
    scratch_root: Path,
    target: str,
) -> list[str]:
    """Copy each ``<src_root>/<name>-<target>/`` dir into ``<scratch_root>/``
    and overwrite its main source with the agent's ``new_main``. Skip
    entries without a submission or without a reference dir. Returns the
    list of benchmark names successfully prepared (autohecbench's
    argument list)."""
    prepared: list[str] = []
    skipped: list[tuple[str, str]] = []

    scratch_root.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        name = entry["name"]
        # Only include tasks the agent actually submitted via submit_solution.
        # Fallback-extracted text (when submit wasn't called) can be garbage
        # like "[max steps reached...]" — we don't want those in the scratch
        # tree since they produce misleading "build failed" CSV gaps.
        if not entry.get("agent_submitted"):
            skipped.append((name, "no submission"))
            continue
        new_main = entry.get("new_main") or ""
        if not new_main:
            skipped.append((name, "empty new_main"))
            continue
        src = src_root / f"{name}-{target}"
        if not src.is_dir():
            skipped.append((name, f"reference missing: {src.name}"))
            continue
        dst = scratch_root / f"{name}-{target}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        target_file = entry.get("target_file") or (
            "main.cu" if target == "cuda" else "main.cpp"
        )
        (dst / target_file).write_text(new_main)
        prepared.append(name)

    if skipped:
        print("Skipped in scratch prep:")
        for n, reason in skipped:
            print(f"  - {n:<24} {reason}")
    return prepared


# ---------------------------------------------------------------------------
# Stage 3: autohecbench runs
# ---------------------------------------------------------------------------

def run_autohecbench(
    names: list[str],
    *,
    bench_dir: Path,
    target: str,
    repeat: int,
    nvidia_sm: int,
    output_csv: Path,
    env_path_prepend: str | None = None,
) -> None:
    """Invoke ``autohecbench.py`` against ``bench_dir`` and dump a CSV to
    ``output_csv``. One row per benchmark, format ``<name>-<target>, t1,
    t2, ...`` (N = ``repeat``)."""
    env = os.environ.copy()
    if env_path_prepend:
        env["PATH"] = env_path_prepend + ":" + env.get("PATH", "")

    cmd = [
        sys.executable, str(AUTOHECBENCH),
        "--yes-prompt",
        "--overwrite",
        "-r", str(repeat),
        "--nvidia-sm", str(nvidia_sm),
        "-b", str(bench_dir),
        "-o", str(output_csv),
        *[f"{n}-{target}" for n in names],
    ]
    run_cmd(cmd, env=env)


def run_compare(old_csv: Path, new_csv: Path, output_md: Path) -> None:
    """Invoke ``autohecbench-compare.py OLD NEW`` and tee the markdown
    speedup table to ``output_md``."""
    cmd = [sys.executable, str(AUTOHECBENCH_CMP), str(old_csv), str(new_csv)]
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    output_md.write_text(proc.stdout)
    print(proc.stdout)


# ---------------------------------------------------------------------------
# Stage 4: merge baseline + candidate CSVs into results.json / results.csv
# ---------------------------------------------------------------------------

def _read_autohecbench_csv(path: Path) -> dict[str, list[float]]:
    """Parse autohecbench's CSV: ``<name>-<target>, v1, v2, ...`` → dict
    keyed by the bare benchmark name (before the first ``-``)."""
    out: dict[str, list[float]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            key = row[0].split("-")[0]
            try:
                out[key] = [float(v) for v in row[1:] if v.strip()]
            except ValueError:
                continue
    return out


def merge_results(
    entries: list[dict[str, Any]],
    *,
    baseline_csv: Path,
    candidate_csv: Path,
    results_json: Path,
    results_csv: Path,
) -> tuple[int, int]:
    """Merge agent metadata with both CSVs, compute per-benchmark speedup.
    Returns (n_timed, n_faster)."""
    baseline = _read_autohecbench_csv(baseline_csv)
    candidate = _read_autohecbench_csv(candidate_csv)

    merged: list[dict[str, Any]] = []
    n_timed = 0
    n_faster = 0
    for entry in entries:
        name = entry["name"]
        b = baseline.get(name, [])
        c = candidate.get(name, [])
        speedup: float | None = None
        if b and c:
            speedup = statistics.mean(b) / statistics.mean(c)
            n_timed += 1
            if speedup > 1.0:
                n_faster += 1
        merged.append({
            "name": name,
            "target": entry["target"],
            "categories": entry.get("categories", []),
            "agent_submitted": entry.get("agent_submitted", False),
            "agent_steps": entry.get("agent_steps"),
            "agent_elapsed_s": entry.get("agent_elapsed_s"),
            "baseline_times": b,
            "candidate_times": c,
            "baseline_mean": statistics.mean(b) if b else None,
            "candidate_mean": statistics.mean(c) if c else None,
            "speedup": speedup,
        })
    results_json.write_text(json.dumps(merged, indent=2))

    with results_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "target", "submitted",
            "baseline_mean", "candidate_mean", "speedup",
        ])
        for m in merged:
            w.writerow([
                m["name"], m["target"], m["agent_submitted"],
                m["baseline_mean"], m["candidate_mean"], m["speedup"],
            ])
    return n_timed, n_faster


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end HeCBench: agent → scratch tree → autohecbench → compare."
    )
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--skip-scratch", action="store_true")
    ap.add_argument("--skip-timing", action="store_true",
                    help="Skip baseline+candidate autohecbench runs")
    ap.add_argument("--cuda-path", default="/usr/local/cuda/bin",
                    help="Prepended to PATH for autohecbench subprocess so "
                         "nvcc is visible (default: /usr/local/cuda/bin)")
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
    bcfg = HeCBenchBenchmarkConfig.from_yaml(bench_yaml)

    # --- Paths ---
    batch_dir = run_dir / "batch"
    batch_dir.mkdir(exist_ok=True)
    scratch_dir = run_dir / "scratch"

    agent_output = run_dir / "agent_output.json"
    baseline_csv = run_dir / "baseline.csv"
    candidate_csv = run_dir / "candidate.csv"
    speedup_md = run_dir / "speedup.md"
    results_json = run_dir / "results.json"
    results_csv = run_dir / "results.csv"

    serial_root = (PROJECT_ROOT / bcfg.serial_root).resolve()
    src_root = (PROJECT_ROOT / bcfg.src_root).resolve()

    print(f"Run directory: {run_dir}")
    print(f"Model:         {acfg.model.name}")
    print(f"Target:        {bcfg.target}  (sm_{bcfg.nvidia_sm}, repeat={bcfg.repeat})")
    print(f"Serial root:   {serial_root}")
    print(f"Src root:      {src_root}")

    # --- Stage 1: Agent ---
    if not args.skip_agent:
        print("\n[1/4] Running agent...")
        adapter_args: dict[str, Any] = {
            "target": bcfg.target,
            "src_root": str(src_root),
        }
        if bcfg.names:
            adapter_args["names"] = bcfg.names
        if bcfg.categories:
            adapter_args["categories"] = bcfg.categories

        system_prompt_args: list[str] = []
        if acfg.system_prompt:
            sp_path = run_dir / "system_prompt.txt"
            sp_path.write_text(acfg.system_prompt)
            system_prompt_args = ["--system-prompt-file", str(sp_path)]

        cmd = [
            sys.executable, "-m", "agent.batch",
            "--adapter", "hecbench",
            "--adapter-args", json.dumps(adapter_args),
            "--prompts", str(serial_root),
            "--output", str(agent_output),
            "--model", acfg.model.name,
            "--base-url", acfg.model.base_url,
            "--api-key", acfg.model.api_key,
            "--temperature", str(acfg.model.temperature),
            "--max-tokens", str(acfg.model.max_tokens),
            *(["--reasoning"] if acfg.model.reasoning else []),
            *system_prompt_args,
            "--max-steps", str(acfg.agent.max_steps),
            "--time-budget", str(acfg.agent.time_budget),
            "--workspace-root", str(batch_dir),
            "--workers", str(acfg.agent.workers),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run_cmd(cmd, cwd=PROJECT_ROOT)
    else:
        print("\n[1/4] Skipped (--skip-agent)")
        if not agent_output.exists():
            sys.exit(f"ERROR: {agent_output} not found")

    entries: list[dict[str, Any]] = json.loads(agent_output.read_text())
    if args.limit:
        entries = entries[:args.limit]

    # --- Stage 2: Scratch tree ---
    if not args.skip_scratch:
        print(f"\n[2/4] Preparing scratch tree at {scratch_dir} ...")
        prepared = prepare_scratch_tree(
            entries,
            src_root=src_root,
            scratch_root=scratch_dir,
            target=bcfg.target,
        )
        print(f"  prepared {len(prepared)} benchmark dirs")
    else:
        print("\n[2/4] Skipped (--skip-scratch)")
        prepared = [p.name.rsplit("-", 1)[0]
                    for p in scratch_dir.iterdir() if p.is_dir()] if scratch_dir.exists() else []

    # --- Stage 3: autohecbench timing ---
    if not args.skip_timing:
        if not prepared:
            print("\n[3/4] No benchmarks prepared, skipping timing.")
        else:
            print(f"\n[3/4a] Timing baseline ({len(prepared)} benchmarks) ...")
            run_autohecbench(
                prepared,
                bench_dir=src_root,
                target=bcfg.target,
                repeat=bcfg.repeat,
                nvidia_sm=bcfg.nvidia_sm,
                output_csv=baseline_csv,
                env_path_prepend=args.cuda_path,
            )
            print(f"\n[3/4b] Timing candidate ...")
            run_autohecbench(
                prepared,
                bench_dir=scratch_dir,
                target=bcfg.target,
                repeat=bcfg.repeat,
                nvidia_sm=bcfg.nvidia_sm,
                output_csv=candidate_csv,
                env_path_prepend=args.cuda_path,
            )
    else:
        print("\n[3/4] Skipped (--skip-timing)")

    # --- Stage 4: Compare + merge ---
    print("\n[4/4] Computing speedup + merging results ...")
    if baseline_csv.exists() and candidate_csv.exists():
        run_compare(baseline_csv, candidate_csv, speedup_md)
    n_timed, n_faster = merge_results(
        entries,
        baseline_csv=baseline_csv,
        candidate_csv=candidate_csv,
        results_json=results_json,
        results_csv=results_csv,
    )

    # --- Summary ---
    submitted = sum(1 for e in entries if e.get("agent_submitted"))
    print(f"\n{'='*60}")
    print(f"  Done! → {run_dir}")
    print(f"{'='*60}")
    print(f"  {len(entries)} tasks  |  submitted={submitted}  |  timed={n_timed}  |  speedup>1.0 = {n_faster}")
    if speedup_md.exists():
        print(f"\n  speedup table: {speedup_md}")


if __name__ == "__main__":
    main()
