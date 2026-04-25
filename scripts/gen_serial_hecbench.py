#!/usr/bin/env python3
"""Generate serial C++ versions of HeCBench benchmarks via LLM.

For each eligible benchmark under ``benchmarks/HeCBench/src/<name>-<model>/``,
send its ``main.cpp`` to an OpenAI-compatible endpoint and ask for a
single-threaded CPU equivalent. Write the result to
``benchmarks/HeCBench/serial/<name>/main.cpp`` (plus ``.meta.json`` with the
args/regex/categories copied from benchmarks.yaml).

The adapter in ``agent/adapters/hecbench.py`` can then be pointed at this
serial directory so the agent's task becomes "parallelize this serial code"
instead of "optimize existing parallel code".

Usage
-----
  uv run python3 scripts/gen_serial_hecbench.py
  uv run python3 scripts/gen_serial_hecbench.py --workers 10 --limit 20
  uv run python3 scripts/gen_serial_hecbench.py --names adam accuracy
  uv run python3 scripts/gen_serial_hecbench.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "http://140.112.90.45:48011/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_YAML = PROJECT_ROOT / "benchmarks" / "HeCBench" / "benchmarks.yaml"
DEFAULT_SRC_ROOT = PROJECT_ROOT / "benchmarks" / "HeCBench" / "src"
DEFAULT_OUT = PROJECT_ROOT / "benchmarks" / "HeCBench" / "serial"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a C++ engineer. You convert parallel GPU/OpenMP code into a
single-threaded CPU-only reference implementation.

Rules:
1. Remove every parallel directive and runtime call:
   - All ``#pragma omp ...`` lines, including ``#pragma omp declare
     target`` / ``end declare target`` and any ``simd`` / ``target`` /
     ``parallel`` / ``for`` variants.
   - CUDA kernel launches (``<<<...>>>``), ``cudaMalloc`` / ``cudaMemcpy``
     / ``cudaFree`` / ``cudaDeviceSynchronize`` and any ``__global__`` /
     ``__device__`` / ``__host__`` / ``__shared__`` qualifiers.
   - HIP equivalents (``hipMalloc``, ``hipLaunchKernelGGL``, …).
   - SYCL constructs (``sycl::queue``, ``parallel_for``, accessors,
     buffers).
   - Calls into ``omp_get_*`` / ``omp_set_*`` runtime helpers (use
     ``std::chrono`` for timing instead).
2. Remove now-dead artifacts so the file looks like real serial code:
   - ``#include <omp.h>``, ``#include <cuda_runtime.h>``,
     ``#include <hip/hip_runtime.h>``, ``#include <sycl/sycl.hpp>``.
   - ``#define NUM_THREADS`` / ``BLOCK_SIZE`` / ``BLOCK_DIM`` /
     ``WARP_SIZE`` / ``TILE_*`` macros that are no longer referenced.
   - Any helper kernels or device functions that nothing calls after the
     conversion.
3. Keep the program's external behavior identical: same CLI arguments,
   same stdout format (including ``PASS``/``FAIL`` and the timing line),
   same RNG seeds, same numerical semantics — ordering, reduction
   formulas, and intermediate precision must match.
4. Compute kernels become plain nested ``for`` loops on the host.
5. The result must compile with ``g++ -O3 -std=c++17`` — no
   ``-fopenmp``, no CUDA/HIP toolchains required.
6. If the input contains multiple source files (separated by
   ``// ===== filename =====`` headers), merge them into one
   self-contained ``main.cpp``.
7. Output ONLY the full replacement ``main.cpp`` inside a single
   ```cpp ... ``` fenced block. No commentary before or after.
"""

_USER_TEMPLATE = """\
Convert this HeCBench benchmark to a serial CPU-only version.

Benchmark: {name} ({model})
CLI args used at test time: {args}

## Original {source_file}
```cpp
{main_src}
```
"""


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:cpp|c\+\+|c)?\s*\n?(.*?)```", re.S)
_HDR_EXTS = {".h", ".hpp", ".hh", ".hxx", ".cuh", ".inc"}
_INCLUDE_DIR = re.compile(r"-I\s*([^\s]+)")
_INT_MAIN = re.compile(r"\bint\s+main\s*\(")
_MK_SOURCE = re.compile(r"^\s*source\s*=\s*(.+)$", re.M)
_MK_OBJ_DEP = re.compile(r"^\s*\S+\.o:\s*(.+)$", re.M)
_SRC_EXTS = (".cpp", ".cc", ".cxx", ".cu", ".c")


def _extract(text: str) -> str:
    m = _FENCE.search(text or "")
    return (m.group(1) if m else text or "").strip()


def _find_sources(bench_dir: Path) -> list[Path]:
    """Return source files for a HeCBench benchmark directory.

    Strategy (in order):
      1. ``main.{cpp,cu,cc,c}`` in the dir.
      2. Makefile ``source = X.cpp Y.cpp`` plus any ``X.o: ../sib/Z.cpp``
         dependency lines (HeCBench shares ``main.cpp`` between siblings
         this way, e.g. ``bwt-omp`` borrows from ``bwt-cuda``).
      3. Any local ``.cpp/.cu/.c`` whose contents include ``int main(``.
    """
    if not bench_dir.is_dir():
        return []
    for cand in ("main.cpp", "main.cu", "main.cc", "main.c"):
        p = bench_dir / cand
        if p.is_file():
            return [p]

    mk = bench_dir / "Makefile"
    if mk.is_file():
        text = mk.read_text(errors="ignore")
        paths: list[Path] = []
        m = _MK_SOURCE.search(text)
        if m:
            for f in m.group(1).split():
                p = (bench_dir / f).resolve()
                if p.is_file():
                    paths.append(p)
        for dm in _MK_OBJ_DEP.finditer(text):
            for tok in dm.group(1).split():
                if not tok.endswith(_SRC_EXTS):
                    continue
                p = (bench_dir / tok).resolve()
                if p.is_file() and p not in paths:
                    paths.append(p)
        if paths:
            return paths

    for ext in _SRC_EXTS:
        for f in sorted(bench_dir.glob(f"*{ext}")):
            try:
                if _INT_MAIN.search(f.read_text(errors="ignore")):
                    return [f]
            except OSError:
                pass
    return []


def _pack_sources(srcs: list[Path], bench_dir: Path) -> tuple[str, str]:
    """Render one or more source files as a single string for the prompt."""
    if len(srcs) == 1:
        return srcs[0].read_text(errors="replace"), srcs[0].name
    parts = []
    for s in srcs:
        try:
            rel = s.relative_to(bench_dir)
        except ValueError:
            rel = Path("..") / s.parent.name / s.name
        parts.append(f"// ===== {rel} =====\n{s.read_text(errors='replace')}")
    return "\n\n".join(parts), ", ".join(s.name for s in srcs)


def _build_entries(
    yaml_path: Path,
    src_model: str,
    src_root: Path,
    names: list[str] | None,
    categories: list[str] | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Walk benchmarks.yaml and resolve source files for each candidate."""
    bench = yaml.safe_load(yaml_path.read_text()) or {}
    entries: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for name in sorted(bench):
        b = bench[name] or {}
        if src_model not in (b.get("models") or []):
            continue
        test = b.get("test") or {}
        if not test:
            skipped.append((name, "no test metadata"))
            continue
        if names is not None and name not in names:
            continue
        if categories is not None:
            if not (set(categories) & set(b.get("categories") or [])):
                continue
        bench_dir = src_root / f"{name}-{src_model}"
        srcs = _find_sources(bench_dir)
        if not srcs:
            skipped.append((name, f"no sources in {bench_dir.name}"))
            continue
        main_src, source_file = _pack_sources(srcs, bench_dir)
        entries.append({
            "name": name,
            "model": src_model,
            "source_dir": str(bench_dir),
            "source_file": source_file,
            "main_src": main_src,
            "args": list(test.get("args") or []),
            "regex": test.get("regex"),
            "timeout": test.get("timeout", 300),
            "categories": list(b.get("categories") or []),
        })
    return entries, skipped


def _collect_sibling_headers(bench_dir: Path) -> list[Path]:
    """Return headers that must be copied alongside main.cpp so the serial
    version is self-contained. Includes:

    1. Every header in the benchmark's own directory.
    2. Every header in directories reached via ``-I../sibling`` in the
       Makefile (HeCBench commonly points ``-omp`` at its ``-cuda`` twin's
       ``reference.h``).
    """
    out: list[Path] = []
    for p in bench_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _HDR_EXTS:
            out.append(p)

    # Parse Makefiles for -I flags → sibling dirs
    for mk_name in ("Makefile", "Makefile.aomp", "Makefile.nvc"):
        mk = bench_dir / mk_name
        if not mk.exists():
            continue
        for inc in _INCLUDE_DIR.findall(mk.read_text()):
            inc_path = (bench_dir / inc).resolve()
            if not inc_path.is_dir() or inc_path == bench_dir.resolve():
                continue
            for p in inc_path.iterdir():
                if p.is_file() and p.suffix.lower() in _HDR_EXTS:
                    out.append(p)
    # De-dup by resolved path
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


# ---------------------------------------------------------------------------
# Per-benchmark worker
# ---------------------------------------------------------------------------

def process_one(
    client: OpenAI,
    model: str,
    entry: dict[str, Any],
    *,
    out_root: Path,
    max_tokens: int,
    temperature: float,
    overwrite: bool,
) -> dict[str, Any]:
    name = entry["name"]
    dst_dir = out_root / name
    dst_main = dst_dir / "main.cpp"
    dst_meta = dst_dir / ".meta.json"

    record = {"name": name, "model": entry["model"]}

    if dst_main.exists() and not overwrite:
        record["status"] = "skip-exists"
        return record

    user = _USER_TEMPLATE.format(
        name=name,
        model=entry["model"],
        args=" ".join(entry.get("args") or []) or "(none)",
        source_file=entry["source_file"],
        main_src=entry["main_src"],
    )

    start = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        return record
    elapsed = time.monotonic() - start

    reply = resp.choices[0].message.content or ""
    code = _extract(reply)
    if not code:
        record["status"] = "empty"
        record["raw_reply_head"] = reply[:500]
        return record

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_main.write_text(code)

    # Copy headers needed for compilation. Flatten into dst_dir so a plain
    # ``g++ -I<dst_dir>`` works.
    headers_copied: list[str] = []
    for hdr in _collect_sibling_headers(Path(entry["source_dir"])):
        dst_hdr = dst_dir / hdr.name
        dst_hdr.write_bytes(hdr.read_bytes())
        headers_copied.append(hdr.name)

    dst_meta.write_text(json.dumps({
        "name": name,
        "source_model": entry["model"],
        "source_dir": entry["source_dir"],
        "source_file": entry["source_file"],
        "args": entry.get("args", []),
        "regex": entry.get("regex"),
        "timeout": entry.get("timeout"),
        "categories": entry.get("categories", []),
        "headers": headers_copied,
        "elapsed_s": round(elapsed, 2),
    }, indent=2))
    record["status"] = "ok"
    record["elapsed_s"] = round(elapsed, 2)
    record["bytes"] = len(code)
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Serialize HeCBench benchmarks via LLM")
    ap.add_argument("--benchmarks-yaml", default=str(DEFAULT_YAML))
    ap.add_argument("--src-model", default="omp",
                    help="Which HeCBench implementation to serialize "
                         "(omp/cuda/hip/sycl). Default omp.")
    ap.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT),
                    help="Root of HeCBench src/ tree.")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N benchmarks")
    ap.add_argument("--names", nargs="*", default=None,
                    help="Only process these benchmark names")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate even if main.cpp already exists")
    args = ap.parse_args()

    entries, skipped = _build_entries(
        Path(args.benchmarks_yaml),
        args.src_model,
        Path(args.src_root),
        args.names,
        args.categories,
    )
    if args.limit:
        entries = entries[: args.limit]
    print(f"loaded {len(entries)} benchmarks "
          f"(skipped {len(skipped)} at scan time)")
    if skipped:
        head = ", ".join(f"{n}({r})" for n, r in skipped[:5])
        more = f" + {len(skipped)-5} more" if len(skipped) > 5 else ""
        print(f"  e.g. {head}{more}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    summary = {"ok": 0, "skip-exists": 0, "empty": 0, "error": 0}
    log_path = out_root / "_gen_log.jsonl"

    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
         log_path.open("a") as logf:
        futs = {
            pool.submit(
                process_one, client, args.model, entry,
                out_root=out_root,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                overwrite=args.overwrite,
            ): entry["name"]
            for entry in entries
        }
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"name": name, "status": "error",
                       "error": f"{type(e).__name__}: {e}"}
            summary[rec["status"]] = summary.get(rec["status"], 0) + 1
            tag = rec["status"]
            extra = ""
            if tag == "ok":
                extra = f" ({rec.get('elapsed_s')}s, {rec.get('bytes')} B)"
            elif tag == "error":
                extra = f" — {rec.get('error','')[:120]}"
            print(f"[{i}/{len(futs)}] {name:<24} {tag}{extra}")
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k:<14} {v}")
    print(f"Output: {out_root}")
    print(f"Log:    {log_path}")


if __name__ == "__main__":
    main()
