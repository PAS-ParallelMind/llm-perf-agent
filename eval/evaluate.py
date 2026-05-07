#!/usr/bin/env python3
"""Evaluate a harness ``agent_output.json`` against ``benchmarks.json``.

Reads
-----
  ``benchmarks.json``    stable corpus: per-problem reference / gen_input /
                         optional checker.
  ``agent_output.json``  harness output: list of ``{id, code, submitted, ...}``
                         (produced by ``agent.batch`` or
                         ``scripts/run_bare.py``).

Per submitted entry, the pipeline is:

  1. **Validate correctness.** Compile reference + candidate, run gen_input
     to produce K inputs, then for each input compare bytes:
       - bytes equal                       → PASS_BYTE
       - bytes differ + checker            → PASS_CHECKER / FAIL
       - bytes differ + non-byte-det       → LLM judge (PASS_LLM / FAIL /
                                              ERROR_LLM)
       - bytes differ + byte-deterministic → FAIL (no LLM fallback)

  2. **Measure speedup** (only if validation is fully-pass). Time the
     reference's compute span (chrono-instrumented to skip I/O), then time
     the candidate's CUDA kernels via ``nsys`` and overall wall-clock.
     Reports ``speedup_e2e`` (ref_wall / cand_wall) and ``speedup_kernel``
     (ref_compute / cand_kernel) — both numerator + denominator exclude
     file I/O.

Output
------
  ``eval_results.json``  list of ``{id, submitted, validation?, speedup?}``
                         aligned 1:1 with the input. Non-submitted entries
                         have ``validation=null, speedup=null``.

Usage
-----
  python eval/evaluate.py \\
      --input runs/agent_v3/agent_output.json \\
      --out   runs/agent_v3/eval_results.json
      [--no-validate | --no-speedup]
      [--no-llm]                       # skip LLM judge fallback
      [--problem P001]                 # restrict to one id
      [--workers N] [--n-runs 3] [--timeout 120]
      [--speedup-all]                  # measure speedup even when
                                       #   validation fails
      [--keep-tempdir]                 # don't auto-delete per-task work
                                       #   dirs (handy for debugging)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def _tempdir(prefix: str, keep: bool):
    """Context manager: tempfile.TemporaryDirectory when keep=False, an
    mkdtemp (never deleted) when keep=True. The kept path is logged to
    stderr on context exit so it's easy to find for debugging."""
    if keep:
        d = tempfile.mkdtemp(prefix=prefix)
        try:
            yield d
        finally:
            print(f"  kept: {d}", file=sys.stderr)
    else:
        with tempfile.TemporaryDirectory(prefix=prefix) as d:
            yield d

EVAL_ROOT = Path(__file__).resolve().parent

DEFAULT_BENCHMARKS = EVAL_ROOT / "benchmarks.json"
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

# Default candidate language. Harness output doesn't carry a language
# field; this benchmark suite is CUDA-targeting.
DEFAULT_CAND_LANG = "cuda"


# ---------------------------------------------------------------------------
# 1. Compile / run helpers
# ---------------------------------------------------------------------------

_NVCC_RUNTIME_CACHE: dict[str, bool] = {}


def _nvcc_runtime_ok(nvcc_path: str) -> bool:
    """Compile a tiny CUDA program and try cudaMalloc; cache result."""
    if nvcc_path in _NVCC_RUNTIME_CACHE:
        return _NVCC_RUNTIME_CACHE[nvcc_path]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cu"
        src.write_text(
            "#include <cuda_runtime.h>\n#include <cstdio>\n"
            "int main(){void*p;cudaError_t e=cudaMalloc(&p,16);"
            "printf(\"%d\",e);return e!=cudaSuccess;}"
        )
        binp = Path(tmp) / "probe"
        cp = subprocess.run(
            [nvcc_path, "-O0", "-arch=" + CUDA_ARCH, str(src), "-o", str(binp)],
            capture_output=True, timeout=60,
        )
        ok = (cp.returncode == 0
              and subprocess.run([str(binp)], capture_output=True,
                                 timeout=10).returncode == 0)
    _NVCC_RUNTIME_CACHE[nvcc_path] = ok
    return ok


def find_compiler(name: str) -> str | None:
    """Find a working compiler. nvcc gets a runtime probe to skip versions
    whose CUDA runtime is too new for the host driver."""
    if name == "nvcc":
        from glob import glob
        cands = sorted(glob("/usr/local/cuda-*/bin/nvcc"), reverse=True)
        cands.append("/usr/local/cuda/bin/nvcc")
        for c in cands:
            if Path(c).is_file() and _nvcc_runtime_ok(c):
                return c
    return shutil.which(name)


def find_nsys() -> str | None:
    from glob import glob
    cands = sorted(glob("/usr/local/cuda-*/bin/nsys"), reverse=True)
    cands.append("/usr/local/cuda/bin/nsys")
    for c in cands:
        if Path(c).is_file():
            return c
    return shutil.which("nsys")


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


def run_gen_input(workdir: Path, gen_block: dict[str, Any],
                  ) -> tuple[bool, str, list[Path]]:
    src = workdir / ("gen_input" + EXT_FOR_LANG.get(
        gen_block.get("language", "python"), ".py"))
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


def gen_one_input_for_speedup(gen_block: dict[str, Any],
                               workdir: Path) -> Path | None:
    """Like run_gen_input but forces --count 1 to keep speedup runs fast."""
    src = workdir / ("gen_input" + EXT_FOR_LANG.get(
        gen_block.get("language", "python"), ".py"))
    src.write_text(gen_block["code"])
    args2 = list(gen_block.get("default_args", []))
    if "--count" in args2:
        args2[args2.index("--count") + 1] = "1"
    p = subprocess.run([sys.executable, str(src), *args2],
                       cwd=workdir, capture_output=True, timeout=120)
    if p.returncode != 0:
        return None
    inp = workdir / "inputs" / "input_0.bin"
    return inp if inp.is_file() else None


# ---------------------------------------------------------------------------
# 2. Checker (Python sandbox)
# ---------------------------------------------------------------------------

def load_checker(checker_block: dict[str, Any] | None):
    """Return ``check(input_bytes, output_bytes) -> {valid, reason}``, or
    None if no checker is defined / loadable."""
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
# 3. LLM judge
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
                      {"role": "user",   "content": user}],
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
# 4. Validation
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
        "pass_rate":       (f"{ok}/{total}"
                            + (f" ({100*ok/total:.1f}%)" if total else "")),
    }


def _all_pass(summary: dict[str, Any]) -> bool:
    """Did every input case pass (byte / checker / llm)?"""
    total = summary["total"]
    if total < 1:
        return False
    return (summary["pass_byte"]
            + summary["pass_checker"]
            + summary["pass_llm"]) == total


def validate_one(prob: dict[str, Any], code: str, language: str, *,
                 check_fn, llm_client, llm_model: str,
                 run_timeout: int, keep_tempdir: bool = False,
                 ) -> dict[str, Any]:
    """Compile + run + judge one candidate. Returns the validation block."""
    description = prob.get("description", prob.get("name", "?"))

    with _tempdir(f"eval_{prob.get('id','prob')}_", keep_tempdir) as tmp_str:
        tmp = Path(tmp_str)

        # Compile reference
        ref_lang = prob["reference"]["language"]
        ref_src  = tmp / f"reference{EXT_FOR_LANG.get(ref_lang, '.cpp')}"
        ref_src.write_text(prob["reference"]["code"])
        ref_bin = tmp / "reference"
        ref_ok, ref_err = compile_source(ref_lang, ref_src, ref_bin)

        gen_ok, gen_err, inputs = (False, "skipped (ref build failed)", [])
        if ref_ok:
            gen_ok, gen_err, inputs = run_gen_input(tmp, prob["gen_input"])

        cand_src = tmp / f"candidate{EXT_FOR_LANG.get(language, '.cpp')}"
        cand_src.write_text(code)
        cand_bin = tmp / "candidate.bin"

        cases: list[CaseResult] = []

        if not ref_ok:
            cases.append(CaseResult(input="(no inputs)", status="BUILD_FAIL_REF",
                                    detail=ref_err[:500]))
        else:
            cand_ok, cand_build_err = compile_source(language, cand_src, cand_bin)
            if not cand_ok:
                cases.append(CaseResult(input="(build)", status="BUILD_FAIL_CAND",
                                        detail=cand_build_err[:500]))
            elif not gen_ok:
                cases.append(CaseResult(input="(no inputs)", status="FAIL_REF",
                                        detail=f"gen_input: {gen_err[:300]}"))
            else:
                cand_name = prob.get("id", "candidate")
                for inp in inputs:
                    ref_out  = tmp / f"_ref_{inp.stem}.bin"
                    cand_out = tmp / f"_cand_{inp.stem}.bin"
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
                        base.status = "PASS_CHECKER" if result["valid"] else "FAIL"
                        if not result["valid"]:
                            base.detail = f"checker: {base.checker_reason}"
                        cases.append(base); continue

                    # 3. LLM judge — only when not byte-deterministic
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
# 5. Speedup measurement
# ---------------------------------------------------------------------------

def instrument_reference(src: str) -> str | None:
    """Inject a chrono timer between the success-path ``fclose(fin)`` and
    the ``fopen(argv[2], ...)`` so the reference prints ``compute_ns=<n>``
    on stderr — the compute-only span, excluding file I/O. Returns None
    if the anchors aren't present."""
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


_COMPUTE_NS_RE = re.compile(r"compute_ns=(\d+)")
_KERN_TOTAL_RE = re.compile(
    r"^\s+(\d+\.\d+)\s+([\d,]+)\s+\d+", re.M  # Time(%), Total Time (ns), Instances
)


def time_run(binary: Path, in_path: Path, out_path: Path, *, n: int = 3,
             ) -> tuple[float, float]:
    """Run ``binary`` ``n`` times, return ``(median wall-ms, median compute-ms)``.

    compute-ms comes from a ``compute_ns=<n>`` line on stderr emitted by
    the instrumented reference; if the binary doesn't emit one (e.g. an
    unmodified candidate), the second value is NaN."""
    walls: list[float] = []
    comps: list[float] = []
    subprocess.run([str(binary), str(in_path), str(out_path)],
                   capture_output=True, timeout=60)  # warm-up
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


def time_nsys_kernel(nsys: str, binary: Path, in_path: Path, out_path: Path,
                     *, tmp: Path) -> float:
    """Run binary under nsys, return SUM of all CUDA kernel times (ms).
    Returns NaN on error or when no kernels were observed."""
    rep = tmp / "report"
    rep.with_suffix(".nsys-rep").unlink(missing_ok=True)
    rep.with_suffix(".sqlite").unlink(missing_ok=True)
    try:
        subprocess.run([
            nsys, "profile", "--force-overwrite=true",
            "--output", str(rep),
            str(binary), str(in_path), str(out_path),
        ], capture_output=True, timeout=180, check=True)
        stats = subprocess.run([
            nsys, "stats", "--report=cuda_gpu_kern_sum",
            str(rep.with_suffix(".nsys-rep")),
        ], capture_output=True, text=True, timeout=60)
    except subprocess.SubprocessError:
        return float("nan")

    total_ns = 0
    for m in _KERN_TOTAL_RE.finditer(stats.stdout):
        try:
            total_ns += int(m.group(2).replace(",", ""))
        except ValueError:
            continue
    return total_ns / 1e6 if total_ns else float("nan")


def measure_speedup(prob: dict[str, Any], code: str, language: str,
                    *, n_runs: int, keep_tempdir: bool = False,
                    ) -> dict[str, Any] | None:
    """Time reference (compute-only) + candidate (kernel-only via nsys).
    Returns None when reference build / input gen / candidate build fails.
    """
    ref_lang = prob["reference"]["language"]
    if ref_lang != "cpp":
        return None  # instrumentation only handles cpp anchors

    nsys = find_nsys()

    with _tempdir(f"speedup_{prob.get('id','p')}_", keep_tempdir) as tmp_str:
        tmp = Path(tmp_str)

        # Reference: instrumented compile
        ref_src = instrument_reference(prob["reference"]["code"]) \
                  or prob["reference"]["code"]
        ref_path = tmp / f"reference{EXT_FOR_LANG.get(ref_lang, '.cpp')}"
        ref_path.write_text(ref_src)
        ref_bin = tmp / "ref"
        ok, _ = compile_source(ref_lang, ref_path, ref_bin)
        if not ok:
            return None

        # One input
        inp = gen_one_input_for_speedup(prob["gen_input"], tmp)
        if inp is None:
            return None

        ref_wall, ref_compute = time_run(
            ref_bin, inp, tmp / "ref_out.bin", n=n_runs)

        # Candidate
        cand_path = tmp / f"candidate{EXT_FOR_LANG.get(language, '.cu')}"
        cand_path.write_text(code)
        cand_bin = tmp / "cand"
        ok, _ = compile_source(language, cand_path, cand_bin)
        if not ok:
            return None

        cand_wall, _ = time_run(cand_bin, inp, tmp / "cand_out.bin", n=n_runs)
        cand_kern = (time_nsys_kernel(nsys, cand_bin, inp, tmp / "cand_out.bin",
                                       tmp=tmp)
                     if (nsys and language == "cuda") else float("nan"))

        def _round(x, p=3):
            return None if (x is None or (isinstance(x, float) and math.isnan(x))) \
                        else round(x, p)

        kern_ok = isinstance(cand_kern, float) and not math.isnan(cand_kern) \
                  and cand_kern > 0
        comp_ok = isinstance(ref_compute, float) and not math.isnan(ref_compute)
        return {
            "ref_wall_ms":     _round(ref_wall),
            "ref_compute_ms":  _round(ref_compute, 4),
            "cand_wall_ms":    _round(cand_wall),
            "cand_kernel_ms":  _round(cand_kern, 4),
            "speedup_e2e":     (round(ref_wall / cand_wall, 3)
                                if cand_wall else None),
            "speedup_kernel":  (round(ref_compute / cand_kern, 3)
                                if (kern_ok and comp_ok) else None),
        }


# ---------------------------------------------------------------------------
# 6. Per-entry orchestration
# ---------------------------------------------------------------------------

def evaluate_one(prob: dict[str, Any], entry: dict[str, Any], *,
                 do_validate: bool, do_speedup: bool, speedup_all: bool,
                 llm_client, llm_model: str,
                 run_timeout: int, n_runs: int,
                 keep_tempdir: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id":         entry["id"],
        "submitted":  bool(entry.get("submitted")),
        "validation": None,
        "speedup":    None,
    }
    if not entry.get("submitted") or not entry.get("code"):
        return out

    code = entry["code"]
    language = (entry.get("metadata") or {}).get("language", DEFAULT_CAND_LANG)

    if do_validate:
        check_fn = load_checker(prob.get("checker"))
        out["validation"] = validate_one(
            prob, code, language,
            check_fn=check_fn,
            llm_client=llm_client, llm_model=llm_model,
            run_timeout=run_timeout, keep_tempdir=keep_tempdir,
        )

    if do_speedup:
        gate_ok = (
            speedup_all
            or (do_validate and out["validation"]
                and _all_pass(out["validation"]["summary"]))
            or (not do_validate)
        )
        if gate_ok:
            out["speedup"] = measure_speedup(
                prob, code, language,
                n_runs=n_runs, keep_tempdir=keep_tempdir,
            )

    return out


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True,
                    help="Harness agent_output.json (list of {id, code, ...})")
    ap.add_argument("--out", required=True,
                    help="Output eval_results.json")
    ap.add_argument("--benchmarks", default=str(DEFAULT_BENCHMARKS))
    ap.add_argument("--problem", default=None,
                    help="Restrict to one problem id (e.g. P002)")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip the correctness check")
    ap.add_argument("--no-speedup", action="store_true",
                    help="Skip the timing/speedup measurement")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM judge fallback (non-byte-det only)")
    ap.add_argument("--speedup-all", action="store_true",
                    help="Measure speedup even when validation isn't all-pass")
    ap.add_argument("--keep-tempdir", action="store_true",
                    help="Keep per-task tempdirs for debugging "
                         "(default: auto-clean)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker threads")
    ap.add_argument("--n-runs", type=int, default=3,
                    help="Median over N timed runs (per direction)")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-binary run timeout (seconds)")
    ap.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    ap.add_argument("--api-key",      default="EMPTY")
    ap.add_argument("--llm-model",    default=DEFAULT_LLM_MODEL)
    args = ap.parse_args()

    do_validate = not args.no_validate
    do_speedup  = not args.no_speedup
    if not (do_validate or do_speedup):
        sys.exit("nothing to do (both --no-validate and --no-speedup set)")

    bench_path = Path(args.benchmarks).resolve()
    in_path    = Path(args.input).resolve()
    out_path   = Path(args.out).resolve()
    if not bench_path.is_file():
        sys.exit(f"benchmarks.json not found: {bench_path}")
    if not in_path.is_file():
        sys.exit(f"agent_output.json not found: {in_path}")

    bench = json.loads(bench_path.read_text())
    problems = bench.get("problems") or {}
    entries  = json.loads(in_path.read_text())
    if not isinstance(entries, list):
        sys.exit("--input: expected a list (harness agent_output.json shape)")

    if args.problem:
        entries = [e for e in entries if e.get("id") == args.problem]
        if not entries:
            sys.exit(f"no entry matched --problem {args.problem!r}")

    # LLM client (lazy)
    llm_client = None
    if do_validate and not args.no_llm:
        try:
            from openai import OpenAI
            llm_client = OpenAI(base_url=args.llm_base_url, api_key=args.api_key)
        except ImportError:
            print("WARN: openai SDK unavailable; falling back to --no-llm semantics")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = [None] * len(entries)  # type: ignore
    print_lock = threading.Lock()

    def _run(idx: int, entry: dict[str, Any]) -> None:
        pid = entry.get("id")
        prob = problems.get(pid)
        if prob is None:
            with print_lock:
                print(f"  [{idx+1}/{len(entries)}] {pid}: NOT in benchmarks.json — skip")
            results[idx] = {"id": pid, "submitted": bool(entry.get("submitted")),
                            "validation": None, "speedup": None}
            return
        r = evaluate_one(
            prob, entry,
            do_validate=do_validate, do_speedup=do_speedup,
            speedup_all=args.speedup_all,
            llm_client=llm_client, llm_model=args.llm_model,
            run_timeout=args.timeout, n_runs=args.n_runs,
            keep_tempdir=args.keep_tempdir,
        )
        results[idx] = r
        v = r["speedup"]
        s = (r["validation"] or {}).get("summary") if r["validation"] else None
        with print_lock:
            v_str = (f"val={s['pass_rate']}" if s else "val=skip")
            sp_str = ("e2e=" + ("None" if not v or v["speedup_e2e"] is None
                                else f"{v['speedup_e2e']}x")
                      + " kern=" + ("None" if not v or v["speedup_kernel"] is None
                                    else f"{v['speedup_kernel']}x")) \
                     if r["speedup"] is not None else "speedup=-"
            print(f"  [{idx+1}/{len(entries)}] {pid:5} {v_str:<28} {sp_str}")

    print(f"evaluating {len(entries)} entries with {args.workers} worker(s)\n")
    if args.workers <= 1:
        for i, e in enumerate(entries):
            _run(i, e)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda ie: _run(*ie), enumerate(entries)))

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    # Final summary
    n = len(results)
    n_pass = sum(1 for r in results
                 if r["validation"] and _all_pass(r["validation"]["summary"]))
    n_sp = sum(1 for r in results if r["speedup"])
    print(f"  {n_pass}/{n} fully pass" + (f"   {n_sp} with speedup" if do_speedup else ""))


if __name__ == "__main__":
    main()
