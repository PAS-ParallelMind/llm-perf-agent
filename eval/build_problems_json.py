#!/usr/bin/env python3
"""Convert eval/benchmarks.json into the harness's problems.json format.

Each ParallelMind eval problem becomes a `{id, prompt, metadata}` entry.
The prompt is the problem's `description` followed by a fenced block
containing the serial C++ reference (so the model has a correct but
non-parallel implementation to translate from).

Output is the input shape expected by `python -m agent.batch --config
run.yaml` (where run.yaml.io.input points at this file).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent
HARNESS_RUNS = EVAL.parent / "runs"
DEFAULT_BENCHMARKS = EVAL / "benchmarks.json"
DEFAULT_OUT = HARNESS_RUNS / "problems.json"


def category_for(pid: str) -> str:
    n = int(pid[1:])
    if n <= 10:  return "pareval"
    if n <= 20: return "hecbench"
    return "original"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default=str(DEFAULT_BENCHMARKS),
                    help=f"input benchmarks.json (default: {DEFAULT_BENCHMARKS})")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output problems.json (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    bench = json.loads(Path(args.benchmarks).read_text())

    out = []
    for pid in sorted(bench["problems"]):
        prob = bench["problems"][pid]
        ref = prob["reference"]
        prompt = (
            f"{prob['description']}\n\n"
            f"## Serial reference (correct, but not parallel)\n\n"
            f"```{ref.get('language', 'cpp')}\n"
            f"{ref['code']}\n"
            f"```\n"
        )
        out.append({
            "id": pid,
            "prompt": prompt,
            "metadata": {
                "name":               prob["name"],
                "category":           category_for(pid),
                "byte_deterministic": bool(prob.get("byte_deterministic")),
                "has_checker":        prob.get("checker") is not None,
            },
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} problems to {args.out}")


if __name__ == "__main__":
    main()
