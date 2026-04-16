"""Build + run helpers for CUDA, MPI, and OpenMP sources.

Each tool compiles the source and immediately runs the binary.
If the build fails, the run is skipped and the compile error is returned.
"""
from __future__ import annotations

import os
import subprocess

from ..workspace import get_root, resolve
from .base import tool

MAX_OUT = 20_000


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
def nvcc_build(src: str, out: str, flags: str = "-O3", args: str = "") -> str:
    cmd = ["nvcc", *flags.split(), _rel(src), "-o", _rel(out)]
    build_out, rc = _run(cmd, timeout=300)
    if rc != 0:
        return build_out
    run_cmd = ["./" + _rel(out), *args.split()] if args else ["./" + _rel(out)]
    run_out, _ = _run(run_cmd)
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
def omp_build(src: str, out: str, compiler: str = "gcc",
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
def mpi_build(src: str, out: str, compiler: str = "mpicc",
              nprocs: int = 2, args: str = "") -> str:
    cmd = [compiler, "-O3", _rel(src), "-o", _rel(out)]
    build_out, rc = _run(cmd)
    if rc != 0:
        return build_out
    run_cmd = ["mpirun", "-n", str(nprocs), "./" + _rel(out), *args.split()] if args else ["mpirun", "-n", str(nprocs), "./" + _rel(out)]
    run_out, _ = _run(run_cmd)
    return build_out + "\n" + run_out
