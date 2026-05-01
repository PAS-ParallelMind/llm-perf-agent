"""HeCBench serial-code generation adapter (agent-driven).

Unlike ``scripts/gen_serial_hecbench.py`` (one-shot LLM call), this
adapter frames serial-code synthesis as an *agentic* task: the LLM gets
all parallel implementations of a benchmark (cuda/omp/hip/sycl, whichever
exist) mounted in its workspace as subdirectories, plus tool access to
``read_file``, ``write_file``, ``cpp_build_and_run``, ``bash`` etc. It
can iterate until the serial code compiles and runs, then call
``submit_solution``.

Workspace layout per task::

    <workspace>/
    ├── cuda/   (mirror of <src_root>/<name>-cuda/   if available)
    ├── omp/    (mirror of <src_root>/<name>-omp/    if available)
    ├── hip/    (mirror of <src_root>/<name>-hip/    if available)
    └── sycl/   (mirror of <src_root>/<name>-sycl/   if available)

Output corpus layout (matches ``scripts/gen_serial_hecbench.py`` so
downstream ``run_hecbench.py`` is unchanged)::

    <out_root>/<name>/
    ├── main.cpp        ← submitted serial source
    ├── <hdr>.h ...     ← any extra files agent wrote with write_file
    └── .meta.json
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from ..workspace import get_root
from .base import AgentResult, AgentTask, BenchmarkAdapter

# Programming-model variants we expose to the agent, in priority order.
_VARIANTS = ("cuda", "omp", "hip", "sycl")

_HDR_EXTS = {".h", ".hpp", ".hh", ".hxx", ".cuh", ".inc"}
_SRC_EXTS = {".cpp", ".cc", ".cxx", ".cu", ".c"}

_USER_TEMPLATE = """\
Produce a single-threaded C++ reference (`main.cpp`) for the HeCBench
benchmark `{name}`.

You have the available parallel implementations mounted as
subdirectories of your workspace:
{variants_listing}

## Task

Write `main.cpp` (in the workspace root) that:

1. Removes every parallel directive and runtime call:
   - All `#pragma omp ...` (including `target`/`simd`/`declare target`).
   - CUDA: kernel launches `<<<...>>>`, `cudaMalloc/Memcpy/Free`,
     `__global__/__device__/__host__/__shared__` qualifiers.
   - HIP: `hipMalloc`, `hipLaunchKernelGGL`, etc.
   - SYCL: `sycl::queue`, `parallel_for`, accessors, buffers.
   - `omp_get_*` / `omp_set_*` runtime helpers — use `std::chrono`
     for timing instead.
2. Removes now-dead artifacts: `#include <omp.h>`, `<cuda_runtime.h>`,
   `<hip/hip_runtime.h>`, `<sycl/sycl.hpp>`; macros like `NUM_THREADS`,
   `BLOCK_SIZE`, `TILE_*`; helper kernels nothing calls anymore.
3. Keeps external behavior identical:
   - Same CLI args: `{args}`
   - Same stdout format (PASS/FAIL marker, the timing line matching
     regex: `{regex}`).
   - Same RNG seeds, ordering, and reduction semantics.
4. Produces a self-contained `main.cpp`. If the original implementation
   had helper headers (e.g. `reference.h`), inline their contents into
   `main.cpp` rather than relying on extra includes.
5. Compiles with `g++ -O3 -std=c++17` — no `-fopenmp`, no CUDA toolchain.

## Verification (required before submitting)

After writing `main.cpp`, call `cpp_build_and_run` to verify:
- It compiles cleanly with `g++ -O3 -std=c++17`.
- Running the binary with args `{args}` produces output containing the
  expected PASS marker and a timing line matching the regex above.

If the build or run fails, fix `main.cpp` and try again. You have up to
{max_steps} steps.

## Submission

When the serial reference compiles and runs correctly, call
`submit_solution(code=...)` with the **full contents of `main.cpp`** as
the `code` argument. The harness will materialize this as the canonical
serial corpus entry for `{name}`.

## Recommended workflow

1. `read_file` the parallel implementation(s) (start with `omp/main.cpp`
   if available, otherwise whichever variant exists).
2. `read_file` any sibling headers (`reference.h`, etc.) that are
   `#include`d.
3. `write_file path="main.cpp" content=...` with your serial draft.
4. `cpp_build_and_run src="main.cpp" out="main" args="{args}"` to verify.
5. Iterate until clean, then `submit_solution`.
"""


def _list_variants(variant_paths: dict[str, str]) -> str:
    if not variant_paths:
        return "    (no parallel implementations found — unusual)"
    lines = []
    for v in _VARIANTS:
        if v in variant_paths:
            lines.append(f"  - `{v}/`  ← {Path(variant_paths[v]).name}")
    return "\n".join(lines)


def _build_instruction(meta: dict[str, Any], max_steps: int) -> str:
    return _USER_TEMPLATE.format(
        name=meta["name"],
        variants_listing=_list_variants(meta["variant_paths"]),
        args=" ".join(meta.get("args") or []) or "(none)",
        regex=meta.get("regex") or "(none)",
        max_steps=max_steps,
    )


_CODE_BLOCK = re.compile(r"```(?:cpp|c\+\+|c)?\s*\n?(.*?)```", re.S)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


class HeCBenchSerialGenAdapter(BenchmarkAdapter):
    """Agent-driven serial generation for HeCBench."""

    def __init__(self) -> None:
        self.skipped: list[tuple[str, str]] = []
        self._max_steps = 15  # overridden in load() if caller passes it

    def load(
        self,
        path: str,
        *,
        src_root: str,
        names: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int | None = None,
        max_steps: int = 15,
        require_variant: str | None = "omp",
        **kwargs: Any,
    ) -> list[AgentTask]:
        """Build one task per benchmark with at least one available variant.

        Parameters
        ----------
        path : str
            Path to ``benchmarks.yaml``.
        src_root : str
            Root of the HeCBench ``src/`` tree.
        names, categories, limit : filters (same semantics as the other
            adapters).
        max_steps : int
            Forwarded into the prompt so the model knows its budget.
        require_variant : str | None
            If set, only emit tasks that have this variant available
            (default ``"omp"``, matching the original gen-serial script).
            Pass ``None`` to accept any variant.
        """
        yaml_path = Path(path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"benchmarks.yaml not found: {yaml_path}")
        src = Path(src_root)
        if not src.is_dir():
            raise FileNotFoundError(f"src_root not found: {src}")

        bench = yaml.safe_load(yaml_path.read_text()) or {}
        self._max_steps = max_steps
        tasks: list[AgentTask] = []
        skipped: list[tuple[str, str]] = []

        for name in sorted(bench):
            b = bench[name] or {}
            test = b.get("test") or {}
            if not test:
                skipped.append((name, "no test metadata"))
                continue
            if names is not None and name not in names:
                continue
            if categories is not None:
                if not (set(categories) & set(b.get("categories") or [])):
                    continue

            variant_paths: dict[str, str] = {}
            for v in _VARIANTS:
                cand = src / f"{name}-{v}"
                if cand.is_dir():
                    variant_paths[v] = str(cand)

            if not variant_paths:
                skipped.append((name, f"no variant dirs under {src.name}/"))
                continue
            if require_variant and require_variant not in variant_paths:
                skipped.append((name, f"missing required variant {require_variant!r}"))
                continue

            meta = {
                "name": name,
                "variant_paths": variant_paths,
                "available_variants": sorted(variant_paths),
                "args": list(test.get("args") or []),
                "regex": test.get("regex"),
                "timeout": test.get("timeout", 300),
                "categories": list(b.get("categories") or []),
                # batch.py copies each entry as a subdir of the workspace
                "seed_subdirs": variant_paths,
            }
            tasks.append(AgentTask(
                id=name,
                instruction=_build_instruction(meta, max_steps=max_steps),
                metadata=meta,
            ))

        if limit:
            tasks = tasks[:limit]

        self.skipped = skipped
        return tasks

    def export(self, results: list[AgentResult], output_path: str) -> None:
        """Materialize the serial corpus.

        ``output_path`` is a *directory* (the corpus root). For each
        submitted result we create ``<output_path>/<name>/`` containing:
          - ``main.cpp`` (extracted from the submission)
          - any extra ``.h`` / ``.hpp`` files the agent wrote in its
            workspace root (so headers it deliberately split out are
            preserved)
          - ``.meta.json`` with args/regex/timeout/categories so the
            downstream eval pipeline can read it directly.

        Also writes ``<output_path>/_agent_log.json`` summarizing the run.
        """
        out_root = Path(output_path)
        out_root.mkdir(parents=True, exist_ok=True)

        log: list[dict[str, Any]] = []
        for r in results:
            entry: dict[str, Any] = {
                "name": r.task_id,
                "submitted": r.submitted,
                "steps": r.steps,
                "elapsed_s": r.elapsed_s,
                "error": r.error,
            }

            if not r.submitted or not r.code:
                entry["status"] = "no-submission"
                log.append(entry)
                continue

            code = _extract_code(r.code)
            if not code:
                entry["status"] = "empty-code"
                log.append(entry)
                continue

            dst = out_root / r.task_id
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "main.cpp").write_text(code)

            # Pull any extra headers the agent placed at the workspace
            # root (not in the seeded variant subdirs) — the agent may
            # have chosen to keep `reference.h` as a separate file.
            extra_headers: list[str] = []
            ws_root = Path(r.metadata.get("workspace") or "")
            if ws_root.is_dir():
                for p in ws_root.iterdir():
                    if (p.is_file()
                            and p.suffix.lower() in _HDR_EXTS
                            and p.name != "main.cpp"):
                        shutil.copy2(p, dst / p.name)
                        extra_headers.append(p.name)

            meta_obj = {
                "name": r.task_id,
                "source_model": "agent-multi",
                "available_variants": r.metadata.get("available_variants", []),
                "args": r.metadata.get("args", []),
                "regex": r.metadata.get("regex"),
                "timeout": r.metadata.get("timeout"),
                "categories": r.metadata.get("categories", []),
                "headers": extra_headers,
                "agent_steps": r.steps,
                "agent_elapsed_s": r.elapsed_s,
            }
            (dst / ".meta.json").write_text(json.dumps(meta_obj, indent=2))

            entry["status"] = "ok"
            entry["headers"] = extra_headers
            log.append(entry)

        (out_root / "_agent_log.json").write_text(json.dumps(log, indent=2))
