"""Build + run helpers for CUDA, MPI, and OpenMP sources.

Each tool compiles the source and immediately runs the binary.
If the build fails, the run is skipped and the compile error is returned.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..cuda import cuda_env as _cuda_env, find_nvcc as _resolve_nvcc
from ..workspace import get_root, resolve
from .base import tool

MAX_OUT = 20_000


def _cuda_runtime_env(nvcc_path: str, base_env: dict | None = None) -> dict:
    """Thin wrapper around ``agent.cuda.cuda_env`` pinned to a specific
    nvcc. Kept for backwards compat with callers that already know which
    nvcc they're using."""
    return _cuda_env(base_env=base_env, nvcc_path=nvcc_path)


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
