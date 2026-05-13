"""Build + run helpers for CUDA, MPI, and OpenMP sources.

Each tool compiles the source and immediately runs the binary.
If the build fails, the run is skipped and the compile error is returned.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..workspace import get_root, resolve
from .base import tool

MAX_OUT = 20_000

# Versioned CUDA install dirs (skip unversioned ``cuda`` symlink so the
# system-admin's choice of default doesn't override the "newest wins"
# semantics). Newest CUDA wins (sorted descending).
_CUDA_PREFIX_GLOBS = ("/usr/local/cuda-*", "/opt/cuda-*")

# CUDA compute capability for the runtime probe. Matches eval/evaluate.py.
_CUDA_PROBE_ARCH = "sm_89"

_NVCC_OK_CACHE: dict[str, bool] = {}


def _nvcc_runtime_ok(nvcc_path: str) -> bool:
    """Compile a tiny CUDA program and try ``cudaMalloc``. Cached.

    Mirrors ``eval/evaluate.py``'s probe so the agent's compile tool can't
    silently fall back to an nvcc whose CUDA runtime is too new for the
    host driver (or, conversely, too old to know about the GPU's arch).
    """
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
            [nvcc_path, "-O0", "-arch=" + _CUDA_PROBE_ARCH,
             str(src), "-o", str(binp)],
            capture_output=True, timeout=60,
        )
        ok = (cp.returncode == 0
              and subprocess.run([str(binp)], capture_output=True,
                                 timeout=10).returncode == 0)
    _NVCC_OK_CACHE[nvcc_path] = ok
    return ok


def _resolve_nvcc() -> str | None:
    """Return a working nvcc.

    Prefer ``/usr/local/cuda-*/bin/nvcc`` (newest first), each verified via
    a tiny compile + cudaMalloc probe. Fall back to ``shutil.which("nvcc")``
    only when none of the local CUDA installs works — the system
    ``/usr/bin/nvcc`` is often an older package that doesn't know about
    newer compute capabilities (e.g. sm_89)."""
    candidates: list[str] = []
    for pat in _CUDA_PREFIX_GLOBS:
        candidates.extend(glob.glob(f"{pat}/bin/nvcc"))
    candidates.sort(reverse=True)
    for c in candidates:
        if _nvcc_runtime_ok(c):
            return c
    return shutil.which("nvcc")


def _cuda_runtime_env(nvcc_path: str, base_env: dict | None = None) -> dict:
    """Return an env dict with CUDA's lib64 prepended to LD_LIBRARY_PATH."""
    env = (base_env or os.environ).copy()
    lib64 = os.path.join(os.path.dirname(os.path.dirname(nvcc_path)), "lib64")
    if os.path.isdir(lib64):
        env["LD_LIBRARY_PATH"] = lib64 + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _run(cmd: list[str], timeout: int = 180, env: dict | None = None) -> tuple[str, int]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(get_root()),
            env=env,
        )
    except FileNotFoundError as e:
        return f"ERROR: {e}", 1
    except subprocess.TimeoutExpired:
        return f"ERROR: timeout after {timeout}s: {' '.join(cmd)}", 1
    out = (p.stdout or "") + (p.stderr or "")
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + "\n... [truncated]"
    return f"$ {' '.join(cmd)}\n[exit={p.returncode}]\n{out}", p.returncode


def _rel(p: str) -> str:
    return str(resolve(p).relative_to(get_root()))


@tool(
    "Compile a CUDA source file and run the binary. "
    "If the build fails, the run is skipped and the compile error is returned.",
    src="Path to .cu file (relative to workspace)",
    out="Output binary path (relative to workspace)",
    flags="Extra nvcc flags (e.g. '-O3 -arch=sm_80')",
    args="Args to pass to the binary (space-separated, default empty)",
)
def nvcc_build_and_run(src: str, out: str, flags: str = "-O3", args: str = "") -> str:
    nvcc = _resolve_nvcc()
    if not nvcc:
        return ("ERROR: nvcc not found on PATH or under /usr/local/cuda*/bin "
                "or /opt/cuda*/bin. Install CUDA toolkit or set PATH.")
    cmd = [nvcc, *flags.split(), _rel(src), "-o", _rel(out)]
    build_out, rc = _run(cmd, timeout=300)
    if rc != 0:
        return build_out
    run_env = _cuda_runtime_env(nvcc)
    run_cmd = ["./" + _rel(out), *args.split()] if args else ["./" + _rel(out)]
    run_out, _ = _run(run_cmd, env=run_env)
    return build_out + "\n" + run_out


@tool(
    "Compile a plain C/C++ source (no OpenMP, no CUDA) and run the binary. "
    "Use this to verify a serial reference compiles and runs cleanly. "
    "If the build fails, the run is skipped and the compile error is returned.",
    src="Path to .c/.cpp file (relative to workspace)",
    out="Output binary path (relative to workspace)",
    compiler="Compiler to use (g++/gcc/clang++)",
    flags="Extra compiler flags (default '-O3 -std=c++17')",
    args="Args to pass to the binary (space-separated, default empty)",
)
def cpp_build_and_run(src: str, out: str, compiler: str = "g++",
                      flags: str = "-O3 -std=c++17", args: str = "") -> str:
    cmd = [compiler, *flags.split(), _rel(src), "-o", _rel(out)]
    build_out, rc = _run(cmd, timeout=180)
    if rc != 0:
        return build_out
    run_cmd = ["./" + _rel(out), *args.split()] if args else ["./" + _rel(out)]
    run_out, _ = _run(run_cmd, timeout=300)
    return build_out + "\n" + run_out


@tool(
    "Compile a C/C++ source with OpenMP and run the binary. "
    "If the build fails, the run is skipped and the compile error is returned.",
    src="Path to .c/.cpp file (relative to workspace)",
    out="Output binary path (relative to workspace)",
    compiler="Compiler to use (gcc/g++/clang)",
    threads="OMP_NUM_THREADS value (default 4)",
    args="Args to pass to the binary (space-separated, default empty)",
)
def omp_build_and_run(src: str, out: str, compiler: str = "gcc",
                      threads: int = 4, args: str = "") -> str:
    cmd = [compiler, "-O3", "-fopenmp", _rel(src), "-o", _rel(out)]
    build_out, rc = _run(cmd)
    if rc != 0:
        return build_out
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    run_cmd = ["./" + _rel(out), *args.split()] if args else ["./" + _rel(out)]
    run_out, _ = _run(run_cmd, env=env)
    return build_out + "\n" + run_out


@tool(
    "Compile an MPI C/C++ source and run the binary with mpirun. "
    "If the build fails, the run is skipped and the compile error is returned.",
    src="Path to source file (relative to workspace)",
    out="Output binary path (relative to workspace)",
    compiler="mpicc or mpicxx",
    nprocs="Number of MPI ranks (default 2)",
    args="Args to pass to the binary (space-separated, default empty)",
)
def mpi_build_and_run(src: str, out: str, compiler: str = "mpicc",
                      nprocs: int = 2, args: str = "") -> str:
    cmd = [compiler, "-O3", _rel(src), "-o", _rel(out)]
    build_out, rc = _run(cmd)
    if rc != 0:
        return build_out
    run_cmd = ["mpirun", "-n", str(nprocs), "./" + _rel(out), *args.split()] if args else ["mpirun", "-n", str(nprocs), "./" + _rel(out)]
    run_out, _ = _run(run_cmd)
    return build_out + "\n" + run_out


def _probe(cmd: list[str], timeout: int = 5) -> str | None:
    """Run ``cmd`` and return the output if it succeeded, else None."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return (p.stdout or p.stderr).strip()


@tool(
    "Inspect the host's parallel-compute setup: available GPUs (name, "
    "compute capability, memory), CUDA/HIP toolchain versions, and host "
    "C/C++ compilers. Call this once before choosing compile flags so you "
    "target the right arch (e.g. ``-arch=sm_89`` vs ``sm_80``). Takes no "
    "arguments."
)
def hardware_info() -> str:
    parts: list[str] = []

    gpus = _probe([
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    parts.append("=== GPUs (nvidia-smi) ===\n" + (gpus or "nvidia-smi unavailable"))

    rocm = _probe(["rocm-smi", "--showproductname"])
    if rocm:
        parts.append("=== GPUs (rocm-smi) ===\n" + rocm)

    nvcc_path = _resolve_nvcc()
    if nvcc_path:
        ver = _probe([nvcc_path, "--version"])
        last = ver.splitlines()[-1] if ver else "(version probe failed)"
        parts.append(f"=== nvcc ===\n{nvcc_path}\n{last}")
    else:
        parts.append("=== nvcc ===\nnot found")

    for comp in ("g++", "clang++", "icpx", "hipcc", "mpicc", "mpicxx"):
        v = _probe([comp, "--version"])
        if v:
            parts.append(f"=== {comp} ===\n" + v.splitlines()[0])

    parts.append(f"=== CPU ===\nlogical cores: {os.cpu_count()}")
    return "\n\n".join(parts)
