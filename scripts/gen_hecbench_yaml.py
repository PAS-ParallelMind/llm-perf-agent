"""Generate HeCBench benchmarks.yaml from CMakeLists.txt + subset.json.

Replaces ``benchmarks/HeCBench/tools/generate_metadata.py`` whose category
parsing is broken (regex bleeds into other CMake commands like
``INCLUDE_DIRS ${SDK_DIR}``).

This generator:
  * Tokenises only the ``add_hecbench_benchmark(...)`` block of each
    ``src/<name>-<model>/CMakeLists.txt`` and stops CATEGORIES at the
    next keyword.
  * Reads ``src/scripts/benchmarks/subset.json`` for ``regex`` / ``args``
    / optional ``binary``.
  * Prints a discrepancy report (yaml-only ghosts vs newly added on disk).

Usage:
    uv run python scripts/gen_hecbench_yaml.py
    uv run python scripts/gen_hecbench_yaml.py --check        # diff only
    uv run python scripts/gen_hecbench_yaml.py -o out.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HECBENCH_ROOT = REPO_ROOT / "benchmarks" / "HeCBench"
SRC_DIR = HECBENCH_ROOT / "src"
SUBSET_JSON = SRC_DIR / "scripts" / "benchmarks" / "subset.json"
DEFAULT_OUT = HECBENCH_ROOT / "benchmarks.yaml"

MODELS = ("cuda", "hip", "omp", "sycl")

# Keyword arguments accepted by add_hecbench_benchmark(). Anything inside
# the call that matches one of these UPPERCASE tokens starts a new section,
# so CATEGORIES can be terminated cleanly without a brittle regex.
CMAKE_KEYWORDS = frozenset({
    "NAME", "MODEL", "SOURCES", "CATEGORIES",
    "INCLUDE_DIRS", "LIBS", "LIBRARIES", "OPTIONS",
    "DEPENDENCIES", "LINK_FLAGS", "LINK_LIBRARIES",
    "COMPILE_FLAGS", "COMPILE_OPTIONS", "DEFINITIONS",
})

_CALL_RE = re.compile(r"add_hecbench_benchmark\s*\((.*?)\)", re.DOTALL)
_COMMENT_RE = re.compile(r"#[^\n]*")


def parse_cmakelists(text: str) -> dict | None:
    """Extract NAME / MODEL / CATEGORIES / SOURCES from a CMakeLists.txt body.

    Returns None if no add_hecbench_benchmark() call is found. CATEGORIES
    is split on whitespace and stops at the next CMake keyword.
    """
    # Strip comments first — some files have `#COMPILE_OPTIONS -ffast-math`
    # inside the call which would otherwise leak into CATEGORIES.
    cleaned = _COMMENT_RE.sub("", text)
    m = _CALL_RE.search(cleaned)
    if not m:
        return None
    tokens = m.group(1).split()
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for tok in tokens:
        if tok in CMAKE_KEYWORDS:
            current = tok
            fields.setdefault(current, [])
        elif current is not None:
            fields[current].append(tok)
    out = {
        "name": fields.get("NAME", [None])[0],
        "model": fields.get("MODEL", [None])[0],
        "sources": fields.get("SOURCES", []),
        "categories": fields.get("CATEGORIES", []),
    }
    if not out["name"] or not out["model"]:
        return None
    return out


def discover() -> dict[str, dict]:
    """Walk src/<name>-<model>/CMakeLists.txt and aggregate by benchmark name."""
    out: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []  # (dir, reason)
    for entry in sorted(SRC_DIR.iterdir()):
        if not entry.is_dir():
            continue
        # split <name>-<model> from directory suffix
        model = next((m for m in MODELS if entry.name.endswith(f"-{m}")), None)
        if model is None:
            continue
        cmake = entry / "CMakeLists.txt"
        if not cmake.is_file():
            skipped.append((entry.name, "no CMakeLists.txt"))
            continue
        parsed = parse_cmakelists(cmake.read_text(errors="replace"))
        if parsed is None:
            skipped.append((entry.name, "no add_hecbench_benchmark()"))
            continue
        # Sanity: declared (NAME, MODEL) should match directory naming.
        # If they disagree, prefer CMake declaration but warn.
        cm_name, cm_model = parsed["name"], parsed["model"]
        dir_name = entry.name[: -(len(model) + 1)]
        if cm_name != dir_name or cm_model != model:
            skipped.append((
                entry.name,
                f"name/model mismatch: dir={dir_name}-{model} cmake={cm_name}-{cm_model}",
            ))
        bench = out.setdefault(cm_name, {"models": set(), "categories": set()})
        bench["models"].add(cm_model)
        bench["categories"].update(parsed["categories"])
    return out, skipped


def load_subset() -> dict[str, dict]:
    if not SUBSET_JSON.is_file():
        return {}
    raw = json.loads(SUBSET_JSON.read_text())
    out: dict[str, dict] = {}
    for name, val in raw.items():
        if not isinstance(val, list):
            continue
        regex = val[0] if len(val) > 0 else ""
        args = val[1] if len(val) > 1 else []
        binary = val[2] if len(val) > 2 else None
        out[name] = {"regex": regex, "args": args, "binary": binary}
    return out


def render_yaml(benchmarks: dict[str, dict], tests: dict[str, dict]) -> str:
    lines = [
        "# HeCBench Benchmark Metadata",
        "#",
        "# Generated by scripts/gen_hecbench_yaml.py from",
        "#   - src/<name>-<model>/CMakeLists.txt   (name, model, categories)",
        "#   - src/scripts/benchmarks/subset.json  (regex, args, binary)",
        "# Do not edit by hand — re-run the generator instead.",
        "#",
        "# Format:",
        "#   benchmark_name:",
        "#     categories: [list of categories]",
        "#     models: [available implementations]",
        "#     test:                       # only present if subset.json has it",
        "#       regex: success-output regex",
        "#       args: [cli args]",
        "#       binary: <name>            # only if not the default 'main'",
        "#       timeout: 300",
        "",
    ]
    for name in sorted(benchmarks):
        b = benchmarks[name]
        cats = sorted(b["categories"]) or ["uncategorised"]
        models = sorted(b["models"])
        lines.append(f"{name}:")
        lines.append(f"  categories: [{', '.join(cats)}]")
        lines.append(f"  models: [{', '.join(models)}]")
        t = tests.get(name)
        if t:
            lines.append("  test:")
            lines.append(f"    regex: '{t['regex'].replace(chr(39), chr(39)*2)}'")
            args_str = ", ".join(f'"{a}"' for a in t["args"]) if t["args"] else ""
            lines.append(f"    args: [{args_str}]")
            if t["binary"] and t["binary"] != "main":
                lines.append(f"    binary: {t['binary']}")
            lines.append("    timeout: 300")
        lines.append("")
    return "\n".join(lines)


def summarise(benchmarks: dict, tests: dict) -> None:
    by_model = {m: 0 for m in MODELS}
    for b in benchmarks.values():
        for m in b["models"]:
            by_model[m] += 1
    with_test = sum(1 for n in benchmarks if n in tests)
    print(f"  benchmarks discovered: {len(benchmarks)}")
    print(f"  with test metadata:    {with_test}")
    for m in MODELS:
        print(f"    {m:5s}: {by_model[m]}")
    orphan_tests = [n for n in tests if n not in benchmarks]
    if orphan_tests:
        print(f"  subset.json entries with no matching benchmark dir: {len(orphan_tests)}")
        for n in orphan_tests[:10]:
            print(f"    - {n}")


def diff_against_shipped(generated: dict, shipped_path: Path) -> int:
    if not shipped_path.is_file():
        print(f"[check] shipped file not found: {shipped_path}")
        return 1
    shipped = yaml.safe_load(shipped_path.read_text()) or {}
    gen_names = set(generated)
    ship_names = set(shipped)
    only_gen = sorted(gen_names - ship_names)
    only_ship = sorted(ship_names - gen_names)
    print(f"[check] shipped: {len(ship_names)}  generated: {len(gen_names)}")
    if only_gen:
        print(f"[check] new on disk (missing from shipped yaml): {len(only_gen)}")
        for n in only_gen[:20]:
            print(f"    + {n}")
        if len(only_gen) > 20:
            print(f"    ... and {len(only_gen) - 20} more")
    if only_ship:
        print(f"[check] in shipped yaml but not on disk: {len(only_ship)}")
        for n in only_ship[:20]:
            print(f"    - {n}")
    # per-name model drift
    drift = []
    for n in gen_names & ship_names:
        gen_m = set(generated[n]["models"])
        ship_m = set(shipped[n].get("models") or [])
        if gen_m != ship_m:
            drift.append((n, sorted(gen_m - ship_m), sorted(ship_m - gen_m)))
    if drift:
        print(f"[check] model-set drift on {len(drift)} benchmarks:")
        for n, added, removed in drift[:15]:
            note = []
            if added: note.append(f"+{added}")
            if removed: note.append(f"-{removed}")
            print(f"    {n}: {' '.join(note)}")
    return 0 if not (only_gen or only_ship or drift) else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                    help=f"Output yaml path (default: {DEFAULT_OUT})")
    ap.add_argument("--check", action="store_true",
                    help="Don't write — diff against shipped benchmarks.yaml")
    ap.add_argument("--stdout", action="store_true",
                    help="Write yaml to stdout instead of a file")
    args = ap.parse_args()

    if not SRC_DIR.is_dir():
        print(f"ERROR: {SRC_DIR} not found", file=sys.stderr)
        return 1

    print(f"Scanning {SRC_DIR.relative_to(REPO_ROOT)}/ ...")
    benchmarks, skipped = discover()
    tests = load_subset()
    summarise(benchmarks, tests)
    if skipped:
        print(f"  skipped {len(skipped)} dir(s):")
        for d, r in skipped[:10]:
            print(f"    - {d}: {r}")

    if args.check:
        return diff_against_shipped(benchmarks, DEFAULT_OUT)

    text = render_yaml(benchmarks, tests)
    if args.stdout:
        sys.stdout.write(text)
        return 0
    args.output.write_text(text)
    try:
        shown = args.output.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"\nwrote {shown}  ({len(text)} bytes)")
    # Validate the produced file actually parses.
    yaml.safe_load(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
