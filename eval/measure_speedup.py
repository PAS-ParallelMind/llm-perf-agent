#!/usr/bin/env python3
"""Measure end-to-end + compute-only speedup for every passing candidate.

For each (problem, candidate) pair where the candidate's validation is
all-pass:
  1. Compile reference (g++) and candidate (nvcc) into a per-task tmp dir.
  2. Run gen_input.py to produce a single input.bin (input_0.bin).
  3. Time the reference with `time.monotonic()` → ref_wall_ms
  4. Time the candidate the same way                 → cand_wall_ms
  5. Run the candidate under `nsys profile --stats=true` and parse
     cuda_gpu_kern_sum total → cand_kernel_ms

Speedups:
  - end_to_end_speedup = ref_wall / cand_wall
  - compute_speedup    = ref_wall / cand_kernel   (cand kernel-only,
                                                    ref's wall ≈ compute
                                                    for our small CPU
                                                    inputs)

Output: parallelmind_harness/runs/legacy/timing.json with per-pid,
per-tag measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EVAL = Path(__file__).resolve().parent
HARNESS_RUNS = EVAL.parent / "runs"
NSYS = "/usr/local/cuda-12.9/bin/nsys"
NVCC = "/usr/local/cuda-12.9/bin/nvcc"
GPP  = "g++"

CUDA_FLAGS = ["-O3", "-std=c++17", "-arch=sm_89", "--extended-lambda"]
GPP_FLAGS  = ["-O3", "-std=c++17"]


def compile_one(src_text: str, ext: str, dst: Path, *, language: str) -> tuple[bool, str]:
    src = dst.parent / f"src{ext}"
    src.write_text(src_text)
    if language == "cuda":
        cmd = [NVCC, *CUDA_FLAGS, str(src), "-o", str(dst)]
    else:
        cmd = [GPP, *GPP_FLAGS, str(src), "-o", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return (p.returncode == 0, p.stderr if p.returncode else "")


def instrument_reference(src: str) -> str | None:
    """Inject a chrono timer around the compute span of a C++ reference.

    Strategy: between the success-path ``fclose(fin)`` and the
    ``fopen(argv[2], ...)`` (both anchors are present in every reference in
    benchmarks.json), emit ``compute_ns=<ns>`` to stderr. Returns the
    instrumented source, or None if anchors aren't found."""
    lines = src.split("\n")
    out_idx = next(
        (i for i, l in enumerate(lines) if "fopen(argv[2]" in l), None
    )
    if out_idx is None:
        return None
    in_idx = next(
        (i for i in range(out_idx - 1, -1, -1) if "fclose(fin)" in lines[i]),
        None,
    )
    if in_idx is None:
        return None

    timer_start = "    auto __pm_t0 = std::chrono::steady_clock::now();"
    timer_end = (
        "    auto __pm_t1 = std::chrono::steady_clock::now();\n"
        "    long long __pm_ns = std::chrono::duration_cast<"
        "std::chrono::nanoseconds>(__pm_t1 - __pm_t0).count();\n"
        "    std::fprintf(stderr, \"compute_ns=%lld\\n\", __pm_ns);"
    )
    out = (
        lines[: in_idx + 1]
        + [timer_start]
        + lines[in_idx + 1 : out_idx]
        + [timer_end]
        + lines[out_idx:]
    )
    if "#include <chrono>" not in src:
        for i, l in enumerate(out):
            if l.startswith("#include"):
                out.insert(i, "#include <chrono>")
                break
    return "\n".join(out)


def gen_one_input(gen_code: str, args: list[str], workdir: Path) -> Path | None:
    src = workdir / "gen.py"
    src.write_text(gen_code)
    # Force --count 1 so we only generate input_0.bin (faster + smaller)
    args2 = list(args)
    if "--count" in args2:
        i = args2.index("--count")
        args2[i + 1] = "1"
    p = subprocess.run([sys.executable, str(src), *args2],
                       cwd=workdir, capture_output=True, timeout=120)
    if p.returncode != 0:
        return None
    inp = workdir / "inputs" / "input_0.bin"
    return inp if inp.is_file() else None


_COMPUTE_NS_RE = re.compile(r"compute_ns=(\d+)")


def time_run(binary: Path, in_path: Path, out_path: Path, *, n: int = 3
             ) -> tuple[float, float]:
    """Run binary n times, return (median wall-ms, median compute-ms).

    compute-ms comes from a `compute_ns=<n>` line on stderr emitted by the
    instrumented reference; if the binary doesn't emit one (e.g. unmodified
    candidate), the second value is nan."""
    walls: list[float] = []
    comps: list[float] = []
    # warm-up once
    subprocess.run([str(binary), str(in_path), str(out_path)],
                   capture_output=True, timeout=60)
    for _ in range(n):
        t0 = time.monotonic()
        p = subprocess.run([str(binary), str(in_path), str(out_path)],
                           capture_output=True, timeout=60)
        if p.returncode != 0:
            return float("nan"), float("nan")
        walls.append((time.monotonic() - t0) * 1000.0)
        m = _COMPUTE_NS_RE.search(p.stderr.decode("utf-8", "ignore"))
        if m:
            comps.append(int(m.group(1)) / 1e6)
    walls.sort()
    comps.sort()
    return (
        walls[len(walls) // 2],
        comps[len(comps) // 2] if comps else float("nan"),
    )


_KERN_TOTAL_RE = re.compile(
    r"^\s+(\d+\.\d+)\s+([\d,]+)\s+\d+", re.M  # Time(%), Total Time (ns), Instances
)


def time_nsys_kernel(binary: Path, in_path: Path, out_path: Path, *,
                      tmp: Path) -> float:
    """Run binary under nsys, return SUM of all CUDA kernel times (ms).
    Returns nan on error."""
    rep = tmp / "report"
    rep.with_suffix(".nsys-rep").unlink(missing_ok=True)
    rep.with_suffix(".sqlite").unlink(missing_ok=True)
    try:
        subprocess.run([
            NSYS, "profile", "--force-overwrite=true",
            "--output", str(rep),
            str(binary), str(in_path), str(out_path),
        ], capture_output=True, timeout=180, check=True)
        stats = subprocess.run([
            NSYS, "stats", "--report=cuda_gpu_kern_sum",
            str(rep.with_suffix(".nsys-rep")),
        ], capture_output=True, text=True, timeout=60)
    except subprocess.SubprocessError:
        return float("nan")

    # Parse table — lines like "  100.0  1,408  1  1408.0  ..."
    total_ns = 0
    for m in _KERN_TOTAL_RE.finditer(stats.stdout):
        try:
            total_ns += int(m.group(2).replace(",", ""))
        except ValueError:
            continue
    return total_ns / 1e6 if total_ns else float("nan")


def passing_pids(s: dict, tag: str) -> list[str]:
    out = []
    for pid in sorted(s["submissions"]):
        cand = (s["submissions"][pid].get("candidates") or {}).get(tag)
        if not cand or "validation" not in cand:
            continue
        summ = cand["validation"]["summary"]
        if summ["total"] > 1 and summ["pass_byte"] + summ["pass_checker"] + summ["pass_llm"] == summ["total"]:
            out.append(pid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(HARNESS_RUNS / "legacy" / "timing.json"))
    ap.add_argument("--n-runs", type=int, default=3)
    args = ap.parse_args()

    bench = json.loads((EVAL / "benchmarks.json").read_text())
    subs = json.loads((EVAL / "submissions.json").read_text())

    bare_pass  = passing_pids(subs, "qwen3-coder_v2")
    agent_pass = passing_pids(subs, "qwen3-coder_agent_v1")
    pids = sorted(set(bare_pass) | set(agent_pass))
    print(f"timing {len(pids)} problems "
          f"(bare passing: {len(bare_pass)}, agent passing: {len(agent_pass)})")

    timing: dict = {}
    for pid in pids:
        prob = bench["problems"][pid]
        ref_lang = prob["reference"]["language"]
        if ref_lang != "cpp":
            print(f"  skip {pid}: reference lang {ref_lang!r} not cpp")
            continue

        with tempfile.TemporaryDirectory(prefix=f"speedup_{pid}_") as tmp_str:
            tmp = Path(tmp_str)
            print(f"\n[{pid}] {prob.get('name','')}")

            # 1. Compile (instrumented) reference
            ref_bin = tmp / "ref"
            ref_src = instrument_reference(prob["reference"]["code"]) \
                      or prob["reference"]["code"]
            ok, err = compile_one(ref_src, ".cpp", ref_bin, language="cpp")
            if not ok:
                print(f"  reference build failed: {err[:200]}")
                continue

            # 2. Generate one input
            inp = gen_one_input(prob["gen_input"]["code"],
                                prob["gen_input"]["default_args"], tmp)
            if not inp:
                print(f"  gen_input failed")
                continue

            # 3. Time reference (wall + compute-only)
            ref_wall, ref_compute = time_run(
                ref_bin, inp, tmp / "ref_out.bin", n=args.n_runs)
            print(f"  ref_wall:  {ref_wall:.2f} ms   "
                  f"ref_compute: {ref_compute:.3f} ms")

            entry = {
                "ref_wall_ms":    round(ref_wall, 3),
                "ref_compute_ms": round(ref_compute, 4)
                                   if ref_compute == ref_compute else None,
            }

            # 4. Time each passing candidate
            for tag in ("qwen3-coder_v2", "qwen3-coder_agent_v1"):
                if pid not in (bare_pass if tag == "qwen3-coder_v2" else agent_pass):
                    continue
                cand_bin = tmp / f"cand_{tag.replace('-','_')}"
                cand_code = subs["submissions"][pid]["candidates"][tag]["code"]
                cand_lang = subs["submissions"][pid]["candidates"][tag]["language"]
                ok, err = compile_one(cand_code, ".cu" if cand_lang == "cuda" else ".cpp",
                                       cand_bin, language=cand_lang)
                if not ok:
                    print(f"  [{tag}] build failed (skip): {err[:150]}")
                    continue
                cand_wall, _ = time_run(cand_bin, inp,
                                         tmp / f"cand_{tag}_out.bin",
                                         n=args.n_runs)
                cand_kern = time_nsys_kernel(cand_bin, inp,
                                              tmp / f"cand_{tag}_out.bin",
                                              tmp=tmp) if cand_lang == "cuda" else float("nan")
                # speedup_kernel uses ref's compute-only time as numerator
                # so both sides exclude file I/O.
                kern_ok = cand_kern == cand_kern and cand_kern > 0
                comp_ok = ref_compute == ref_compute
                entry[tag] = {
                    "wall_ms":         round(cand_wall, 3),
                    "kernel_ms":       round(cand_kern, 3),
                    "speedup_e2e":     round(ref_wall / cand_wall, 3)
                                        if cand_wall else None,
                    "speedup_kernel":  round(ref_compute / cand_kern, 3)
                                        if (kern_ok and comp_ok) else None,
                }
                print(f"  [{tag}] wall={cand_wall:.2f}ms  "
                      f"kernel={cand_kern:.3f}ms  "
                      f"e2e={entry[tag]['speedup_e2e']}x  "
                      f"kernel={entry[tag]['speedup_kernel']}x")
            timing[pid] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(timing, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
