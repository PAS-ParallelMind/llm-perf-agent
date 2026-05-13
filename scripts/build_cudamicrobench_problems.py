#!/usr/bin/env python3
"""Build problems.json entries from a local CUDAMicroBench checkout."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".md",
    ".sh",
}
TEXT_NAMES = {"Makefile"}
SKIP_DIRS = {".git", "lib"}
COMMON_ALLOWLIST = {
    "exception.h",
    "helper_cuda.h",
    "helper_functions.h",
    "helper_image.h",
    "helper_string.h",
    "helper_timer.h",
}


DEFAULT_SELECTION = [
    "HDOverlap",
    "CoMem_AXPY",
    "BankRedux",
    "Shmem",
    "ReadOnlyMem_1D_Texture",
]


def is_text_file(path: Path) -> bool:
    return path.name in TEXT_NAMES or path.suffix in TEXT_SUFFIXES


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def discover_benchmarks(root: Path) -> list[Path]:
    makefiles = []
    for path in root.rglob("Makefile"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        makefiles.append(path.parent)
    return sorted(makefiles, key=lambda p: p.relative_to(root).as_posix())


def problem_id(root: Path, bench_dir: Path) -> str:
    return bench_dir.relative_to(root).as_posix().replace("/", "__")


def filter_benchmarks(root: Path, bench_dirs: list[Path], include: list[str]) -> list[Path]:
    if not include:
        return bench_dirs
    wanted = set(include)
    matched = [p for p in bench_dirs if problem_id(root, p) in wanted]
    missing = sorted(wanted - {problem_id(root, p) for p in matched})
    if missing:
        raise SystemExit(f"unknown CUDAMicroBench ids: {', '.join(missing)}")
    return matched


def collect_seed_files(root: Path, bench_dir: Path, include_common: bool) -> dict[str, str]:
    seed_files: dict[str, str] = {}
    roots = [bench_dir]
    if include_common and needs_common(bench_dir) and (root / "Common").is_dir():
        roots.append(root / "Common")

    for base in roots:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or not is_text_file(path):
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                continue
            if base.name == "Common" and path.name not in COMMON_ALLOWLIST:
                continue
            seed_files[path.relative_to(root).as_posix()] = read_text(path)
    return seed_files


def editable_sources(root: Path, bench_dir: Path) -> list[str]:
    suffixes = {".c", ".cc", ".cpp", ".cu", ".h", ".hpp"}
    return [
        path.relative_to(root).as_posix()
        for path in sorted(bench_dir.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]


def needs_common(bench_dir: Path) -> bool:
    for path in bench_dir.rglob("*"):
        if not path.is_file() or not is_text_file(path):
            continue
        text = read_text(path)
        if "../Common" in text or "../../Common" in text or "helper_cuda" in text:
            return True
    return False


def output_files(root: Path, bench_dir: Path) -> list[str]:
    names = []
    for path in sorted(bench_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".txt" or path.name in {"result.txt", "results.txt", "testResults.txt"}:
            names.append(path.relative_to(root).as_posix())
    return names


def infer_profile_command(bench_dir: Path) -> str | None:
    test_sh = bench_dir / "test.sh"
    if test_sh.exists():
        for line in read_text(test_sh).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("nvprof "):
                line = line.removeprefix("nvprof ").strip()
            return line

    makefile = bench_dir / "Makefile"
    if makefile.exists():
        match = re.search(r"\bnvcc\b.*?\s-o\s+(\S+)", read_text(makefile))
        if match:
            return f"./{match.group(1)}"
    return None


def strip_nvprof(line: str) -> str:
    """Return the benchmark invocation from a test.sh line.

    Several CUDAMicroBench tests use nvprof, which does not run on newer
    NVIDIA GPUs. For validation we only need the benchmark program output, so
    keep the executable and its arguments.
    """
    parts = shlex.split(line)
    if not parts:
        return ""
    if parts[0] != "nvprof":
        return shlex.join(parts)
    for i, part in enumerate(parts[1:], start=1):
        if part.startswith("./") or "/" in part:
            return shlex.join(parts[i:])
    return shlex.join(parts[1:])


def infer_test_command(bench_dir: Path) -> str | None:
    test_sh = bench_dir / "test.sh"
    if not test_sh.exists():
        return None
    commands = []
    for line in read_text(test_sh).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        commands.append(strip_nvprof(line))
    return " && ".join(cmd for cmd in commands if cmd) or None


def make_prompt(
    rel_dir: str,
    test_cmd: str | None,
    profile_cmd: str | None,
    sources: list[str],
) -> str:
    source_list = "\n".join(f"- `{p}`" for p in sources) or "- inspect `Makefile` and source files"
    test_line = f"`cd {rel_dir} && {test_cmd}`" if test_cmd else "no `test.sh`; use the profile command and build result"
    profile_line = f"`cd {rel_dir} && {profile_cmd}`" if profile_cmd else "`cd {rel_dir} && make`, then inspect the produced executable"
    source_arg = sources[0] if sources else f"{rel_dir}/<source-file>"
    run_arg = profile_cmd or "./<executable>"
    test_cmd_arg = test_cmd or "./<executable>"
    return f"""CUDAMicroBench task `{rel_dir}`.
Goal: improve performance while preserving correctness and the existing command-line interface.

Editable source files:
{source_list}

Required workflow:
1. Call `hardware_info` once.
2. Read `Makefile` and the editable source files. Do not rename files and do not invent source filenames.
3. Build exactly with `cd {rel_dir} && make`.
4. Profile exactly with {profile_line}. Use cuda_profile and the returned profile_information to identify the bottleneck. If the binary is missing, build first rather than changing filenames.
5. Edit only the listed source files, preserving program behavior and command-line interface. Make changes justified by profile evidence.
6. Rebuild with `cd {rel_dir} && make`.
7. Validate with {test_line}.
8. Use `cuda_profile` for focused before/after timing. Keep `timeline="auto"` so the profiler can add Nsight Systems timeline facts when the ncu facts indicate they are useful.
9. Call `submit_solution` with a concise summary of changed files and measured before/after result. If no measured improvement is found, say so clearly.

Tool call templates to follow:
- `hardware_info()`
- `read_file(path="{rel_dir}/Makefile")`
- `read_file(path="{source_arg}")`
- `cuda_profile(workdir="{rel_dir}", build_command="make", run_command="{run_arg}", repeats=3, timeline="auto")`
- Edit only the listed source files with `edit_file` or `write_file`.
- `bash(command="cd {rel_dir} && {test_cmd_arg}")`
- `submit_solution(code="Changed <files>; build/test passed; measured result: <brief summary>.")`

Do not use `nvcc_build_and_run` for this benchmark unless the Makefile is unusable. The evaluator reads the modified workspace files; `submit_solution` is only the completion signal."""


def build_problem(root: Path, bench_dir: Path, include_common: bool) -> dict[str, object]:
    rel_dir = bench_dir.relative_to(root).as_posix()
    test_cmd = infer_test_command(bench_dir)
    profile_cmd = infer_profile_command(bench_dir)
    sources = editable_sources(root, bench_dir)
    return {
        "id": problem_id(root, bench_dir),
        "prompt": make_prompt(rel_dir, test_cmd, profile_cmd, sources),
        "seed_files": collect_seed_files(root, bench_dir, include_common),
        "metadata": {
            "benchmark": "CUDAMicroBench",
            "source_dir": rel_dir,
            "build_command": f"cd {rel_dir} && make",
            "profile_command": f"cd {rel_dir} && {profile_cmd}" if profile_cmd else None,
            "test_command": f"cd {rel_dir} && {test_cmd}" if test_cmd else None,
            "reference_outputs": output_files(root, bench_dir),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        help="Only emit these benchmark ids, e.g. HDOverlap CoMem_AXPY",
    )
    parser.add_argument(
        "--default-selection",
        action="store_true",
        help=f"Emit the recommended starter set: {' '.join(DEFAULT_SELECTION)}",
    )
    parser.add_argument("--no-common", action="store_true", help="do not include Common/ helper files")
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    if not (root / "README.md").exists():
        raise SystemExit(f"{root} does not look like a CUDAMicroBench checkout")

    include = DEFAULT_SELECTION if args.default_selection else args.include
    bench_dirs = filter_benchmarks(root, discover_benchmarks(root), include)
    problems = [
        build_problem(root, bench_dir, include_common=not args.no_common)
        for bench_dir in bench_dirs
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
