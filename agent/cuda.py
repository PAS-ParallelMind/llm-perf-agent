"""Shared CUDA toolchain helpers — single source of truth for nvcc
resolution + env injection across:

* ``agent/tools/parallel.py``  — ``nvcc_build_and_run`` tool
* ``agent/tools/bash.py``      — generic shell exec
* ``agent/tools/profile.py``   — profiling tool
* ``eval/evaluate.py``         — offline evaluator

Without this, the agent's compile environment can silently drift from
the evaluator's — e.g. thrust+cub behaviour differs between cuda-12.9
and cuda-13.0, so picking different nvcc paths produces code that
builds on one toolchain but not the other.

The resolution policy is "newest versioned install, runtime-probed":
glob ``/usr/local/cuda-*/bin/nvcc`` + ``/opt/cuda-*/bin/nvcc``, sort
descending, return the first one that successfully compiles AND runs a
tiny cudaMalloc probe (skips nvccs whose CUDA runtime is too new for
the host driver). Falls back to ``shutil.which('nvcc')`` only when no
local install works.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

# Compute capability for the probe; must match the GPU the host has.
# RTX 4090 / Ada Lovelace = sm_89.
CUDA_PROBE_ARCH = "sm_89"

# Versioned dirs only — skip the unversioned ``cuda`` symlink so that
# the system admin's choice of default doesn't override "newest wins".
_CUDA_NVCC_GLOBS = ("/usr/local/cuda-*/bin/nvcc", "/opt/cuda-*/bin/nvcc")

_NVCC_OK_CACHE: dict[str, bool] = {}


def nvcc_runtime_ok(nvcc_path: str) -> bool:
    """Compile a tiny CUDA program and try ``cudaMalloc``. Cached."""
    if nvcc_path in _NVCC_OK_CACHE:
        return _NVCC_OK_CACHE[nvcc_path]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cu"
        src.write_text(
            "#include <cuda_runtime.h>\n#include <cstdio>\n"
            "int main(){void*p;cudaError_t e=cudaMalloc(&p,16);"
            "printf(\"%d\",e);return e!=cudaSuccess;}"
        )
        binp = Path(tmp) / "probe"
        cp = subprocess.run(
            [nvcc_path, "-O0", "-arch=" + CUDA_PROBE_ARCH,
             str(src), "-o", str(binp)],
            capture_output=True, timeout=60,
        )
        ok = (cp.returncode == 0
              and subprocess.run([str(binp)], capture_output=True,
                                 timeout=10).returncode == 0)
    _NVCC_OK_CACHE[nvcc_path] = ok
    return ok


def find_nvcc() -> str | None:
    """Return a working nvcc, or None if nothing works."""
    cands: list[str] = []
    for pat in _CUDA_NVCC_GLOBS:
        cands.extend(glob(pat))
    cands.sort(reverse=True)
    for c in cands:
        if Path(c).is_file() and nvcc_runtime_ok(c):
            return c
    return shutil.which("nvcc")


def cuda_env(base_env: dict | None = None,
             nvcc_path: str | None = None) -> dict:
    """Return an env dict with ``CUDA_HOME`` / ``PATH`` / ``LD_LIBRARY_PATH``
    set so subprocess invocations of nvcc/runtime-linked binaries pick
    the same toolchain as ``find_nvcc()``.

    ``nvcc_path`` lets a caller pin to a specific nvcc (skip resolution).
    """
    env = (base_env or os.environ).copy()
    nvcc = nvcc_path or find_nvcc()
    if not nvcc:
        return env
    bin_dir = os.path.dirname(nvcc)
    prefix = os.path.dirname(bin_dir)
    env["CUDA_HOME"] = prefix
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    lib64 = os.path.join(prefix, "lib64")
    if os.path.isdir(lib64):
        env["LD_LIBRARY_PATH"] = lib64 + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env
