"""HeCBench benchmark adapter — serial → parallel task framing.

Input:  a directory of serial CPU implementations (produced by
        ``scripts/gen_serial_hecbench.py``), each at
        ``<serial_root>/<name>/main.cpp`` with an accompanying
        ``.meta.json`` holding args / regex / timeout from
        ``benchmarks.yaml``.
Task:   given the serial source, produce a parallel implementation for
        the configured target (``cuda`` / ``omp`` / ``hip`` / ``sycl``).
        The model returns a full replacement source (e.g. ``main.cu``
        for CUDA) inside a fenced block.
Output: a JSON list of {name, target, serial_*, new_main, agent_*}
        consumed by ``scripts/run_hecbench.py`` for building + scoring.

Target-specific notes (CUDA kernel launch syntax, OMP pragmas, etc.)
are injected into the prompt via ``_TARGET_NOTES`` so the model knows
which toolchain to target without being shown the reference answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import AgentResult, AgentTask, BenchmarkAdapter

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_USER_TEMPLATE = """\
Parallelize the following serial HeCBench benchmark using {target}.

Name: {name}
Target: {target}
Categories: {categories}
Output file: {target_file}

## Task
Write a replacement ``{target_file}`` that parallelizes the compute
kernels using {target}. Keep the program functionally identical:

* Accept the same command-line arguments: ``{args}``
* Still print ``PASS`` (or whatever correctness marker the program uses)
  on the verification path.
* Still emit a timing line matching this regex (the harness extracts
  the runtime from it):
      {regex}
* Leave host setup/teardown, RNG seeding, and the reference verification
  path unchanged — only the compute hotspots move to the device.

{target_notes}

Return the full ``{target_file}`` inside a single ```cpp ... ``` code
block, then call ``submit_solution`` with that code.

## Serial reference ({serial_file})
```cpp
{serial_src}
```
"""

_TARGET_NOTES: dict[str, str] = {
    "cuda": """\
Target toolchain: ``nvcc -O3 -std=c++17``.
* Use the CUDA runtime API: ``cudaMalloc`` / ``cudaMemcpy`` /
  ``cudaFree`` for device memory, ``<<<grid, block>>>`` for kernel
  launches. ``#include <cuda_runtime.h>``.
* Pick reasonable block/grid dimensions (e.g. 256 threads per block;
  grid = ceil(N / 256)).
* Wrap the timed region around only the GPU compute + required
  memcpy, matching the serial version's timing boundaries.""",
    "omp": """\
Target toolchain: OpenMP for CPU (``g++ -O3 -std=c++17 -fopenmp``).
* Add ``#pragma omp parallel for`` (with ``reduction`` / ``schedule``
  clauses where appropriate) to the compute loops.
* ``#include <omp.h>``.""",
    "hip": """\
Target toolchain: HIP (``hipcc -O3``).
* Use ``hipMalloc`` / ``hipMemcpy`` / ``hipFree``;
  ``hipLaunchKernelGGL`` or ``<<<...>>>`` for kernel launches.
* ``#include <hip/hip_runtime.h>``.""",
    "sycl": """\
Target toolchain: SYCL (``icpx -fsycl`` or ``clang++ -fsycl``).
* Create a ``sycl::queue``, dispatch compute via ``q.parallel_for``,
  use buffers+accessors or USM for data movement.
* ``#include <sycl/sycl.hpp>``.""",
}

# File extension the model should output, by target.
_TARGET_EXT: dict[str, str] = {
    "cuda": ".cu",
    "omp":  ".cpp",
    "hip":  ".cpp",
    "sycl": ".cpp",
}


def _build_instruction(entry: dict[str, Any]) -> str:
    target = entry["target"]
    return _USER_TEMPLATE.format(
        name=entry["name"],
        target=target,
        categories=", ".join(entry.get("categories", [])) or "(unspecified)",
        target_file=entry["target_file"],
        args=" ".join(entry.get("args", [])) or "(none)",
        regex=entry.get("regex", "(none)"),
        target_notes=_TARGET_NOTES.get(target, f"Target toolchain: {target}"),
        serial_file=entry["serial_file"],
        serial_src=entry["serial_src"],
    )


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:cpp|c\+\+|c|cu|cuda)?\s*\n?(.*?)```", re.S)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class HeCBenchAdapter(BenchmarkAdapter):
    """Serial → parallel adapter for HeCBench."""

    def load(
        self,
        path: str,
        *,
        target: str = "cuda",
        src_root: str | None = None,
        names: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[AgentTask]:
        """Scan ``path`` (a serial-root directory) and build one AgentTask
        per benchmark.

        Parameters
        ----------
        path : str
            Directory containing ``<name>/main.cpp`` + ``<name>/.meta.json``
            pairs (as produced by ``scripts/gen_serial_hecbench.py``).
        target : str
            Parallel model to produce — ``cuda`` (default), ``omp``,
            ``hip``, or ``sycl``.
        src_root : str, optional
            Root of the original HeCBench ``src/`` tree. When provided,
            the existence of ``<src_root>/<name>-<target>/`` is recorded
            in task metadata as ``reference_dir`` (used by the eval
            pipeline for baseline timing).
        names : list[str], optional
            Allow-list of benchmark names.
        categories : list[str], optional
            Keep only benchmarks whose categories intersect this list.
        limit : int, optional
            Keep only the first N benchmarks after filtering.
        """
        serial_root = Path(path)
        if not serial_root.is_dir():
            raise FileNotFoundError(f"serial root not found: {serial_root}")

        target_ext = _TARGET_EXT.get(target, ".cpp")

        tasks: list[AgentTask] = []
        skipped: list[tuple[str, str]] = []

        for sub in sorted(p for p in serial_root.iterdir() if p.is_dir()):
            if sub.name.startswith("_") or sub.name.startswith("."):
                continue

            main_path = sub / "main.cpp"
            meta_path = sub / ".meta.json"
            if not main_path.exists() or not meta_path.exists():
                skipped.append((sub.name, "missing main.cpp or .meta.json"))
                continue

            meta = json.loads(meta_path.read_text())
            name = meta.get("name", sub.name)

            if names is not None and name not in names:
                continue
            if categories is not None:
                if not set(categories) & set(meta.get("categories") or []):
                    continue

            reference_dir: str | None = None
            if src_root:
                cand = Path(src_root) / f"{name}-{target}"
                if cand.is_dir():
                    reference_dir = str(cand)

            entry = {
                "name": name,
                "target": target,
                "serial_dir": str(sub),
                "serial_file": main_path.name,
                "serial_src": main_path.read_text(),
                "target_file": f"main{target_ext}",
                "reference_dir": reference_dir,
                "categories": meta.get("categories", []),
                "args": list(meta.get("args") or []),
                "regex": meta.get("regex"),
                "timeout": meta.get("timeout", 300),
                # Pre-seed the agent's workspace with these files so it can
                # actually compile + run (reference.h and any siblings).
                "seed_dir": str(sub),
            }
            tasks.append(AgentTask(
                id=name,
                instruction=_build_instruction(entry),
                metadata=entry,
            ))

        if limit:
            tasks = tasks[:limit]

        self.skipped = skipped
        return tasks

    def export(self, results: list[AgentResult], output_path: str) -> None:
        """Write results as JSON. Each entry keeps the original metadata
        (minus the bulky ``serial_src`` which is on disk) and gains
        ``new_main`` + ``agent_*`` fields."""
        out: list[dict[str, Any]] = []
        for r in results:
            entry = dict(r.metadata)
            entry.pop("serial_src", None)
            entry["new_main"] = _extract_code(r.code) if r.code else ""
            entry["agent_steps"] = r.steps
            entry["agent_elapsed_s"] = r.elapsed_s
            entry["agent_submitted"] = r.submitted
            if r.error:
                entry["agent_error"] = r.error
            out.append(entry)

        Path(output_path).write_text(json.dumps(out, indent=2))
