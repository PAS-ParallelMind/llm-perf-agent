#!/usr/bin/env python3
"""Validator — two-file JSON benchmark layout.

Reads:
    benchmarks.json   stable: problems / reference / gen_input / checker
    submissions.json  mutable: candidates (LLM-generated code) + validation results

Validation flow per (problem, candidate, input)
-----------------------------------------------
    1. Run reference + candidate; compare bytes.
    2. Bytes equal                       → PASS_BYTE
    3. Bytes differ, problem has checker → run checker on candidate output:
            valid                        → PASS_CHECKER
            invalid                      → FAIL
    4. No checker → run LLM judge:
            EQUIVALENT                   → PASS_LLM
            DIFFERENT                    → FAIL
            UNDETERMINED / unreachable   → ERROR_LLM

All source files (reference / gen_input / candidate / checker) are
materialized to a tempdir at runtime and deleted after.

Usage
-----
    uv run --no-project --with openai python3 validate.py
    uv run --no-project python3 validate.py --no-llm
    uv run --no-project python3 validate.py --problem P002 --candidate serial_reverse
    uv run --no-project python3 validate.py --in-place              # write to submissions.json
    uv run --no-project python3 validate.py --output report.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent

DEFAULT_BENCHMARKS  = EVAL_ROOT / "benchmarks.json"
DEFAULT_SUBMISSIONS = EVAL_ROOT / "submissions.json"
DEFAULT_LLM_BASE_URL = "http://140.112.90.46:8001/v1"
DEFAULT_LLM_MODEL    = "/mnt/disk2/elton7318/vllm/weight/Qwen3-Coder-30B-A3B-Instruct-FP8"

CUDA_ARCH = "sm_89"

EXT_FOR_LANG = {
    "cpp":    ".cpp",
    "c":      ".c",
    "omp":    ".cpp",
    "cuda":   ".cu",
    "hip":    ".cpp",
    "python": ".py",
}

BUILD_CMD_FOR_LANG = {
    "cpp":  ["g++", "-O3", "-std=c++17", "{src}", "-o", "{bin}"],
    "c":    ["gcc", "-O3", "{src}", "-o", "{bin}"],
    "omp":  ["g++", "-O3", "-std=c++17", "-fopenmp", "{src}", "-o", "{bin}"],
    "cuda": ["nvcc", "-O3", "-std=c++17", "-arch=" + CUDA_ARCH,
             "--extended-lambda", "{src}", "-o", "{bin}"],
}


# ---------------------------------------------------------------------------
# Compile / run helpers
# ---------------------------------------------------------------------------

def find_compiler(name: str) -> str | None:
    # For nvcc: try /usr/local/cuda-*/bin/nvcc, but skip versions whose
    # CUDA runtime is too new for the host driver. We test each candidate
    # by compiling a one-line program and checking the runtime works.
    if name == "nvcc":
        from glob import glob
        cands = sorted(glob("/usr/local/cuda-*/bin/nvcc"), reverse=True)
        cands += ["/usr/local/cuda/bin/nvcc"]
        for c in cands:
            if not Path(c).is_file():
                continue
            if _nvcc_runtime_ok(c):
                return c
    p = shutil.which(name)
    if p:
        return p
    return None


_NVCC_RUNTIME_CACHE: dict[str, bool] = {}


def _nvcc_runtime_ok(nvcc_path: str) -> bool:
    """Compile a tiny CUDA program and try cudaMalloc; cache result."""
    if nvcc_path in _NVCC_RUNTIME_CACHE:
        return _NVCC_RUNTIME_CACHE[nvcc_path]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cu"
        src.write_text(
            "#include <cuda_runtime.h>\n#include <cstdio>\n"
            "int main(){void*p;cudaError_t e=cudaMalloc(&p,16);"
            "printf(\"%d\",e);return e!=cudaSuccess;}"
        )
        binp = Path(tmp) / "probe"
        cp = subprocess.run(
            [nvcc_path, "-O0", "-arch=sm_89", str(src), "-o", str(binp)],
            capture_output=True, timeout=60,
        )
        ok = (cp.returncode == 0
              and subprocess.run([str(binp)], capture_output=True,
                                 timeout=10).returncode == 0)
    _NVCC_RUNTIME_CACHE[nvcc_path] = ok
    return ok


def compile_source(language: str, src: Path, bin_path: Path) -> tuple[bool, str]:
    if language not in BUILD_CMD_FOR_LANG:
        return False, f"unsupported language: {language!r}"
    template = BUILD_CMD_FOR_LANG[language]
    compiler = find_compiler(template[0])
    if not compiler:
        return False, f"compiler {template[0]!r} not found"
    cmd = [compiler] + [
        s.replace("{src}", str(src)).replace("{bin}", str(bin_path))
        for s in template[1:]
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "(no compiler output)").strip()
    return True, ""


@dataclass
class RunResult:
    ok: bool
    elapsed_s: float
    rc: int
    stderr: str
    output_bytes: bytes | None


def run_binary(binary: Path, in_path: Path, out_path: Path,
               timeout: int) -> RunResult:
    out_path.unlink(missing_ok=True)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [str(binary), str(in_path), str(out_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            return RunResult(False, elapsed, proc.returncode,
                             proc.stderr.strip(), None)
        try:
            data = out_path.read_bytes()
        except OSError as e:
            return RunResult(False, elapsed, 0,
                             f"could not read output.bin: {e}", None)
        return RunResult(True, elapsed, 0, proc.stderr.strip(), data)
    except subprocess.TimeoutExpired:
        return RunResult(False, time.monotonic() - t0, -1,
                         f"timeout after {timeout}s", None)


def run_gen_input(workdir: Path, gen_block: dict[str, Any]) -> tuple[bool, str, list[Path]]:
    src = workdir / ("gen_input" + EXT_FOR_LANG.get(gen_block.get("language", "python"), ".py"))
    src.write_text(gen_block["code"])
    args = list(gen_block.get("default_args", []))
    proc = subprocess.run([sys.executable, str(src)] + args,
                          cwd=workdir, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return False, f"gen_input failed: {proc.stderr.strip()}", []
    inputs_dir = workdir / "inputs"
    if not inputs_dir.is_dir():
        return False, f"gen_input did not produce inputs/ under {workdir}", []
    inputs = sorted(inputs_dir.glob("input_*.bin"))
    if not inputs:
        return False, "gen_input produced no input_*.bin", []
    return True, "", inputs


# ---------------------------------------------------------------------------
# Checker (Python sandbox)
# ---------------------------------------------------------------------------

def load_checker(checker_block: dict[str, Any] | None):
    """Return a callable check(input_bytes, output_bytes) -> {valid, reason},
    or None if no checker is defined / loadable."""
    if not checker_block:
        return None
    if checker_block.get("language") != "python":
        return None
    code = checker_block.get("code") or ""
    namespace: dict[str, Any] = {}
    try:
        exec(compile(code, "<checker>", "exec"), namespace)
    except Exception as e:
        print(f"WARN: checker failed to load: {e}", file=sys.stderr)
        return None
    fn = namespace.get("check")
    if not callable(fn):
        print("WARN: checker has no `check` function", file=sys.stderr)
        return None
    return fn


def run_checker(check_fn, input_bytes: bytes, output_bytes: bytes) -> dict:
    try:
        r = check_fn(input_bytes, output_bytes)
        if not isinstance(r, dict) or "valid" not in r:
            return {"valid": False, "reason": f"checker returned bad shape: {r!r}"}
        r.setdefault("reason", "")
        return r
    except Exception as e:
        return {"valid": False, "reason": f"checker raised: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

LLM_SYSTEM = """\
You evaluate whether two implementations of the same problem produced
SEMANTICALLY EQUIVALENT outputs, even when their bytes differ.

The reference output comes from a serial CPU implementation that we
trust as the spec. The candidate output is from a parallel/optimized
implementation. They diverge at byte level — your job is to decide
whether that divergence is harmless (e.g. floating-point rounding,
permitted tie-breaking, ordering invariants) or a real bug.

Respond ONLY with a JSON object inside a fenced ```json ... ``` block,
with these fields:
  - verdict: one of "EQUIVALENT", "DIFFERENT", "UNDETERMINED"
  - reasoning: one or two sentences explaining the call

Be strict. If the problem statement implies a unique answer (counts,
sums, sorted-by-unique-key), bytes that differ ARE a bug. Only rule
EQUIVALENT when you can name the specific freedom the spec allows."""

LLM_USER_TEMPLATE = """\
## Problem
{description}

## Input case
file: {input_name}  ({input_size} bytes)

## Reference output
size: {ref_size} bytes  hex: {ref_hex}  int32: {ref_int}

## Candidate output ({candidate_name})
size: {cand_size} bytes  hex: {cand_hex}  int32: {cand_int}

The two outputs differ at byte level. Are they semantically equivalent?"""

_JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.S)


def _decode_view(data: bytes, max_bytes: int = 64) -> tuple[str, str]:
    head = data[:max_bytes]
    hex_view = head.hex(" ", 4)
    n_int32 = min(len(head) // 4, 8)
    if n_int32 == 0:
        return hex_view, "(too short)"
    import struct
    ints = struct.unpack(f"<{n_int32}i", head[:n_int32 * 4])
    return hex_view, str(list(ints))


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def llm_judge(client, model: str, *,
              description: str, input_name: str, input_size: int,
              ref_bytes: bytes, cand_bytes: bytes,
              candidate_name: str) -> dict:
    ref_hex, ref_int   = _decode_view(ref_bytes)
    cand_hex, cand_int = _decode_view(cand_bytes)
    user = LLM_USER_TEMPLATE.format(
        description=description, input_name=input_name, input_size=input_size,
        ref_size=len(ref_bytes), ref_hex=ref_hex, ref_int=ref_int,
        cand_size=len(cand_bytes), cand_hex=cand_hex, cand_int=cand_int,
        candidate_name=candidate_name,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": LLM_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as e:
        return {"verdict": "UNDETERMINED",
                "reasoning": f"LLM call failed: {type(e).__name__}: {e}"}
    raw = resp.choices[0].message.content or ""
    parsed = _extract_json(raw)
    if not parsed or "verdict" not in parsed:
        return {"verdict": "UNDETERMINED",
                "reasoning": f"could not parse LLM reply: {raw[:300]}"}
    parsed.setdefault("reasoning", "")
    return parsed


# ---------------------------------------------------------------------------
# Per-candidate validation
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    input: str
    status: str
    ref_elapsed_s: float = 0.0
    cand_elapsed_s: float = 0.0
    ref_bytes: int = 0
    cand_bytes: int = 0
    detail: str = ""
    checker_valid: bool | None = None
    checker_reason: str | None = None
    llm_verdict: str | None = None
    llm_reasoning: str | None = None


def _summarize(cases: list[CaseResult]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.status] = counts.get(c.status, 0) + 1
    total = len(cases)
    ok = (counts.get("PASS_BYTE", 0)
          + counts.get("PASS_CHECKER", 0)
          + counts.get("PASS_LLM", 0))
    return {
        "total":           total,
        "pass_byte":       counts.get("PASS_BYTE",       0),
        "pass_checker":    counts.get("PASS_CHECKER",    0),
        "pass_llm":        counts.get("PASS_LLM",        0),
        "fail":            counts.get("FAIL",            0),
        "fail_ref":        counts.get("FAIL_REF",        0),
        "fail_cand":       counts.get("FAIL_CAND",       0),
        "error_llm":       counts.get("ERROR_LLM",       0),
        "build_fail_ref":  counts.get("BUILD_FAIL_REF",  0),
        "build_fail_cand": counts.get("BUILD_FAIL_CAND", 0),
        "pass_rate":       (f"{ok}/{total}" + (f" ({100*ok/total:.1f}%)" if total else "")),
    }


def validate_candidate(prob: dict[str, Any], cand_name: str, cand: dict[str, Any], *,
                       check_fn, llm_client, llm_model: str,
                       run_timeout: int) -> dict[str, Any]:
    """Compile + run + judge one candidate. Returns the validation block."""
    description = prob.get("description", prob.get("name", "?"))

    with tempfile.TemporaryDirectory(prefix=f"eval_{prob.get('id','prob')}_") as tmp_str:
        tmp = Path(tmp_str)

        # Compile reference
        ref_lang = prob["reference"]["language"]
        ref_src  = tmp / f"reference{EXT_FOR_LANG.get(ref_lang, '.cpp')}"
        ref_src.write_text(prob["reference"]["code"])
        ref_bin = tmp / "reference"
        ref_ok, ref_err = compile_source(ref_lang, ref_src, ref_bin)

        # Generate inputs
        gen_ok, gen_err, inputs = (False, "skipped (ref build failed)", [])
        if ref_ok:
            gen_ok, gen_err, inputs = run_gen_input(tmp, prob["gen_input"])

        # Compile candidate
        cand_lang = cand["language"]
        cand_src  = tmp / f"{cand_name}{EXT_FOR_LANG.get(cand_lang, '.cpp')}"
        cand_src.write_text(cand["code"])
        cand_bin = tmp / f"{cand_name}.bin"

        cases: list[CaseResult] = []

        if not ref_ok:
            cases.append(CaseResult(input="(no inputs)", status="BUILD_FAIL_REF",
                                    detail=ref_err[:500]))
        else:
            cand_ok, cand_build_err = compile_source(cand_lang, cand_src, cand_bin)
            if not cand_ok:
                cases.append(CaseResult(input="(build)", status="BUILD_FAIL_CAND",
                                        detail=cand_build_err[:500]))
            elif not gen_ok:
                cases.append(CaseResult(input="(no inputs)", status="FAIL_REF",
                                        detail=f"gen_input: {gen_err[:300]}"))
            else:
                for inp in inputs:
                    ref_out  = tmp / f"_ref_{inp.stem}.bin"
                    cand_out = tmp / f"_{cand_name}_{inp.stem}.bin"
                    rr = run_binary(ref_bin, inp, ref_out, timeout=run_timeout)
                    base = CaseResult(input=inp.name, status="?",
                                      ref_elapsed_s=round(rr.elapsed_s, 4))
                    if not rr.ok:
                        base.status = "FAIL_REF"
                        base.detail = f"ref rc={rr.rc}: {rr.stderr[:200]}"
                        cases.append(base); continue
                    ref_data = rr.output_bytes or b""
                    base.ref_bytes = len(ref_data)

                    cr = run_binary(cand_bin, inp, cand_out, timeout=run_timeout)
                    base.cand_elapsed_s = round(cr.elapsed_s, 4)
                    if not cr.ok:
                        base.status = "FAIL_CAND"
                        base.detail = f"cand rc={cr.rc}: {cr.stderr[:200]}"
                        cases.append(base); continue
                    cand_data = cr.output_bytes or b""
                    base.cand_bytes = len(cand_data)

                    # 1. byte-equal
                    if ref_data == cand_data:
                        base.status = "PASS_BYTE"
                        cases.append(base); continue

                    # 2. checker (if defined)
                    if check_fn is not None:
                        input_bytes = inp.read_bytes()
                        result = run_checker(check_fn, input_bytes, cand_data)
                        base.checker_valid  = result["valid"]
                        base.checker_reason = (result.get("reason") or "")[:500]
                        if result["valid"]:
                            base.status = "PASS_CHECKER"
                        else:
                            base.status = "FAIL"
                            base.detail = f"checker: {base.checker_reason}"
                        cases.append(base); continue

                    # 3. LLM judge — but ONLY when the problem is *not*
                    #    byte-deterministic. For byte-deterministic
                    #    problems, mismatched bytes mean the candidate
                    #    is wrong; LLM fallback would just add noise.
                    if prob.get("byte_deterministic", True):
                        base.status = "FAIL"
                        base.detail = ("byte mismatch on byte-deterministic "
                                       "problem (no checker / no LLM)")
                        cases.append(base); continue
                    if llm_client is None:
                        base.status = "FAIL"
                        base.detail = "byte mismatch; LLM disabled"
                        cases.append(base); continue
                    judge = llm_judge(
                        llm_client, llm_model,
                        description=description, input_name=inp.name,
                        input_size=inp.stat().st_size,
                        ref_bytes=ref_data, cand_bytes=cand_data,
                        candidate_name=cand_name,
                    )
                    base.llm_verdict   = judge.get("verdict")
                    base.llm_reasoning = (judge.get("reasoning") or "")[:500]
                    if base.llm_verdict == "EQUIVALENT":
                        base.status = "PASS_LLM"
                    elif base.llm_verdict == "DIFFERENT":
                        base.status = "FAIL"
                    else:
                        base.status = "ERROR_LLM"
                    cases.append(base)

        return {
            "summary": _summarize(cases),
            "cases":   [asdict(c) for c in cases],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmarks",  default=str(DEFAULT_BENCHMARKS))
    ap.add_argument("--submissions", default=str(DEFAULT_SUBMISSIONS))
    ap.add_argument("--problem",     default=None,
                    help="Restrict to one problem id (e.g. P002).")
    ap.add_argument("--candidate",   default=None,
                    help="Restrict to one candidate name.")
    ap.add_argument("--no-llm",      action="store_true")
    ap.add_argument("--base-url",    default=DEFAULT_LLM_BASE_URL)
    ap.add_argument("--api-key",     default="EMPTY")
    ap.add_argument("--model",       default=DEFAULT_LLM_MODEL)
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-binary run timeout (seconds).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Run candidates in parallel with N worker threads.")
    ap.add_argument("--in-place",    action="store_true",
                    help="Write validation back into submissions.json.")
    ap.add_argument("--output",      default=None,
                    help="Write a separate report JSON.")
    args = ap.parse_args()

    bench_path = Path(args.benchmarks).resolve()
    sub_path   = Path(args.submissions).resolve()
    if not bench_path.is_file():
        sys.exit(f"benchmarks.json not found: {bench_path}")
    if not sub_path.is_file():
        sys.exit(f"submissions.json not found: {sub_path}")

    bench = json.loads(bench_path.read_text())
    subs  = json.loads(sub_path.read_text())
    problems    = bench.get("problems")    or {}
    submissions = subs.get("submissions") or {}
    if not problems:
        sys.exit("no problems in benchmarks.json")

    # LLM client (lazy)
    llm_client = None
    if not args.no_llm:
        try:
            from openai import OpenAI
            llm_client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        except ImportError:
            print("WARN: openai SDK unavailable; falling back to --no-llm semantics")

    selected_ids = [pid for pid in problems
                    if args.problem is None or pid == args.problem]
    if not selected_ids:
        sys.exit(f"no problem id matched filter {args.problem!r}")

    # Build the (problem, candidate) job list up front so we can dispatch
    # in parallel.
    jobs = []
    for pid in selected_ids:
        prob = problems[pid]
        sub  = submissions.setdefault(pid, {"candidates": {}})
        check_fn = load_checker(prob.get("checker"))
        candidates = sub.get("candidates") or {}
        for cand_name, cand in candidates.items():
            if args.candidate and cand_name != args.candidate:
                continue
            byte_det = prob.get("byte_deterministic", True)
            if check_fn:
                judge = "checker"
            elif (not byte_det) and llm_client is not None:
                judge = "llm"
            else:
                judge = "byte-only"
            jobs.append({
                "pid": pid, "prob": prob, "name": cand_name,
                "cand": cand, "check_fn": check_fn, "judge": judge,
            })

    print(f"Validating {len(jobs)} (problem × candidate) job(s) "
          f"with {args.workers} worker(s)\n")
    if not jobs:
        print("  (no candidates to validate)")

    print_lock = __import__("threading").Lock()

    def _run(job: dict) -> dict:
        v = validate_candidate(
            job["prob"], job["name"], job["cand"],
            check_fn=job["check_fn"],
            llm_client=llm_client, llm_model=args.model,
            run_timeout=args.timeout,
        )
        job["cand"]["validation"] = v
        s = v["summary"]
        with print_lock:
            print(f"  [{job['pid']}] {job['name']:<22} judge={job['judge']:<10} "
                  f"{s['pass_rate']:<14} "
                  f"byte={s['pass_byte']} checker={s['pass_checker']} "
                  f"llm={s['pass_llm']} fail={s['fail']} "
                  f"build_fail_cand={s['build_fail_cand']}")
        return job

    if args.workers <= 1:
        for job in jobs:
            _run(job)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(_run, jobs))
    print()

    # Persistence
    if args.in_place:
        sub_path.write_text(json.dumps(subs, indent=2))
        print(f"wrote submissions.json (in-place): {sub_path}")
    if args.output:
        out_path = Path(args.output).resolve()
        report = {
            "summary": _summarize_global(submissions, selected_ids),
            "problems": {
                pid: {c_name: c.get("validation")
                      for c_name, c in (submissions.get(pid, {}).get("candidates") or {}).items()}
                for pid in selected_ids
            },
        }
        out_path.write_text(json.dumps(report, indent=2))
        print(f"wrote separate report: {out_path}")
    if not args.in_place and not args.output:
        print("(no --in-place or --output: results NOT persisted)")

    # Exit code
    bad = 0
    for pid in selected_ids:
        for cand in (submissions.get(pid, {}).get("candidates") or {}).values():
            for c in (cand.get("validation") or {}).get("cases", []):
                if c.get("status") not in ("PASS_BYTE", "PASS_CHECKER", "PASS_LLM"):
                    bad += 1
    sys.exit(0 if bad == 0 else 1)


def _summarize_global(submissions: dict, ids: list[str]) -> dict:
    all_cases: list[CaseResult] = []
    for pid in ids:
        for cand in (submissions.get(pid, {}).get("candidates") or {}).values():
            for c in (cand.get("validation") or {}).get("cases", []):
                all_cases.append(CaseResult(**{
                    k: v for k, v in c.items()
                    if k in CaseResult.__dataclass_fields__
                }))
    return _summarize(all_cases)


if __name__ == "__main__":
    main()
