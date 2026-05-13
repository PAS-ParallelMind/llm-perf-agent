#!/usr/bin/env python3
"""Render a per-run comparison table as a PNG.

Consumes the new ``eval/evaluate.py`` output schema:
  - <run-dir>/eval_results.json   list[{id, submitted, validation?, speedup?}]
  - <run-dir>/agent_output.json   list[{id, submitted, steps, elapsed_s, ...}]
                                  (optional — used for steps/elapsed metadata)

Each ``--run`` becomes one block of columns ([status pill | rate | e2e/kern]).

Usage
-----
  python visualize_tool/render_comparison.py \\
      runs/agent_v3 \\
      runs/nemotron-3-nano-omni-30b-fp8_v1 \\
      [--labels qwen3-coder nemotron] \\
      [--title "ParallelMind Eval — Qwen3 vs Nemotron"] \\
      [--out runs/_compare.png]   # default: <first-run>/comparison.png

Output PNG defaults under the first run directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HARNESS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARKS = HARNESS_ROOT / "eval" / "benchmarks.json"

# Status colours
C_PASS    = "#a3d8a3"
C_FAIL    = "#f0928b"
C_BUILD   = "#f0b870"
C_PARTIAL = "#f5d870"
C_NA      = "#dddddd"


# ---------------------------------------------------------------------------
# Per-entry classification + formatting helpers
# ---------------------------------------------------------------------------

def category_for(pid: str) -> str:
    n = int(pid[1:])
    if n <= 10: return "pareval"
    if n <= 20: return "hecbench"
    return "original"


def short_name(name: str) -> str:
    for pre, repl in (("pareval_", "p_"),
                      ("hecbench_", "h_"),
                      ("original_", "o_")):
        if name.startswith(pre):
            return repl + name[len(pre):]
    return name


def status_label(entry: dict | None) -> tuple[str, str]:
    """Map an eval_results.json entry → (label, colour)."""
    if not entry:
        return ("—", C_NA)
    if not entry.get("submitted"):
        return ("—", C_NA)
    v = entry.get("validation")
    if not v:
        return ("—", C_NA)
    s = v["summary"]
    total = s["total"]
    if total <= 1:
        # only one synthetic "case" → build / run failure marker
        cases = v.get("cases", [])
        st = cases[0]["status"] if cases else ""
        if "BUILD" in st:
            return ("BUILD", C_BUILD)
        return ("FAIL", C_FAIL)
    passed = s["pass_byte"] + s["pass_checker"] + s["pass_llm"]
    if passed == total:
        return ("PASS", C_PASS)
    if passed == 0:
        return ("FAIL", C_FAIL)
    return ("PARTIAL", C_PARTIAL)


def passed_total(entry: dict | None) -> tuple[int, int] | None:
    if not entry or not entry.get("validation"):
        return None
    s = entry["validation"]["summary"]
    if s["total"] <= 1:
        return None
    return (s["pass_byte"] + s["pass_checker"] + s["pass_llm"], s["total"])


def _num(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if x >= 100: return f"{x:.0f}×"
    if x >= 10:  return f"{x:.1f}×"
    if x >= 0.1: return f"{x:.2f}×"
    if x > 0:    return f"{x:.3f}×"
    return f"{x:.2f}×"


def fmt_speedup(entry: dict | None) -> str:
    if not entry or not entry.get("speedup"):
        return "—"
    sp = entry["speedup"]
    return f"{_num(sp.get('speedup_e2e'))} / {_num(sp.get('speedup_kernel'))}"


def geomean(xs):
    xs = [x for x in xs
          if isinstance(x, (int, float)) and not math.isnan(x) and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path, label: str | None = None) -> dict:
    """Return a dict with keys: label, eval (by id), agent (by id)."""
    eval_path = run_dir / "eval_results.json"
    if not eval_path.is_file():
        raise SystemExit(f"missing eval_results.json under {run_dir}")
    ev = {e["id"]: e for e in json.loads(eval_path.read_text())}

    agent_path = run_dir / "agent_output.json"
    ag = {e["id"]: e for e in json.loads(agent_path.read_text())} \
        if agent_path.is_file() else {}

    return {"label": label or run_dir.name, "dir": run_dir,
            "eval": ev, "agent": ag}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(runs: list[dict], bench: dict, out_path: Path,
           title: str) -> None:
    pids = sorted(bench["problems"].keys())
    n_runs = len(runs)

    # Build per-row data
    rows = []
    for pid in pids:
        prob = bench["problems"][pid]
        per_run = []
        for r in runs:
            ent = r["eval"].get(pid)
            lbl, col = status_label(ent)
            pt = passed_total(ent)
            per_run.append({
                "lbl": lbl, "col": col,
                "rate": f"{pt[0]}/{pt[1]}" if pt else "",
                "sp":   fmt_speedup(ent),
            })

        det = "byte" if prob.get("byte_deterministic") else "checker"
        # Use first run's agent metadata for steps/elapsed
        ag0 = runs[0]["agent"].get(pid, {})
        steps = ag0.get("steps", "")
        elapsed_s = ag0.get("elapsed_s")
        elapsed = ""
        if isinstance(elapsed_s, (int, float)):
            m, s = divmod(int(elapsed_s), 60)
            elapsed = f"{m}m{s:02d}s"

        rows.append({
            "pid": pid,
            "name": short_name(prob["name"]),
            "type": det,
            "per_run": per_run,
            "steps": steps,
            "elapsed": elapsed,
        })

    # ---- Column widths ----
    # Fixed: pid, name, type
    # Per run: status pill, rate, speedup
    # Trailing (1st run only): steps, elapsed
    fixed_w = [("pid", 0.045), ("name", 0.18), ("type", 0.055)]
    per_run_w = [(None, 0.055), (None, 0.045), (None, 0.085)]   # 0.185 each
    trail_w = [("steps", 0.045), ("elapsed", 0.055)] if n_runs <= 2 else []

    cols: list[tuple[str | None, float]] = []
    cols.extend(fixed_w)
    for ri in range(n_runs):
        cols.extend(per_run_w)
    cols.extend(trail_w)

    # Normalize widths to sum to 1.0
    total_w = sum(w for _, w in cols)
    cols = [(lbl, w / total_w) for lbl, w in cols]

    xs = []
    x = 0.0
    for _, w in cols:
        xs.append(x); x += w

    # ---- Figure dims ----
    n = len(rows)
    fig_w = 9.5 + max(0, n_runs - 2) * 1.5
    row_h_in = 0.21
    header_h = 0.55
    title_h = 0.55
    summary_h = 1.20 + 0.30 * n_runs
    fig_h = title_h + header_h + n * row_h_in + summary_h + 0.25

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    ax = fig.add_axes([0.012, 0.015, 0.976, 0.97])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def y_top_of(i_row):
        return 1 - (title_h + header_h + i_row * row_h_in) / fig_h

    # ---- Title ----
    ax.text(0.0, 1 - 0.20/fig_h, title,
            fontsize=14, fontweight="bold", va="top")
    subtitle = "  vs  ".join(r["label"] for r in runs)
    ax.text(0.0, 1 - 0.42/fig_h, subtitle,
            fontsize=9, color="#666", va="top")

    # Legend (top right)
    legend_items = [("PASS", C_PASS), ("FAIL", C_FAIL),
                    ("BUILD", C_BUILD), ("PARTIAL", C_PARTIAL),
                    ("n/a", C_NA)]
    lx = 1.0 - 0.40
    for label, col in legend_items:
        ax.add_patch(Rectangle((lx, 1 - 0.30/fig_h), 0.030, 0.018,
                               facecolor=col, edgecolor="none",
                               transform=ax.transAxes))
        ax.text(lx + 0.034, 1 - 0.295/fig_h, label,
                fontsize=8, va="center")
        lx += 0.078

    # ---- Header row ----
    y_hdr = 1 - (title_h + 0.05) / fig_h
    y_hdr_bot = 1 - (title_h + header_h) / fig_h
    ax.add_patch(Rectangle((0, y_hdr_bot), 1, y_hdr - y_hdr_bot,
                           facecolor="#f4f4f4", edgecolor="none"))

    # Group header (per run label) above the columns
    for ri, r in enumerate(runs):
        col_start = 3 + ri * 3
        col_end   = col_start + 2
        x_l = xs[col_start]
        x_r = xs[col_end] + cols[col_end][1]
        ax.text((x_l + x_r) / 2, y_hdr - 0.012, r["label"],
                fontsize=8.5, fontweight="bold",
                ha="center", va="top", color="#333")
        # Three column labels under the group label
        for ci, sub in enumerate(("status", "rate", "e2e / kern")):
            j = col_start + ci
            ax.text(xs[j] + cols[j][1]/2,
                    y_hdr_bot + (y_hdr - y_hdr_bot) * 0.30, sub,
                    fontsize=7.5, color="#666",
                    ha="center", va="center")

    # Fixed-column labels
    for j, (lbl, w) in enumerate(cols):
        if lbl is None:
            continue
        ax.text(xs[j] + w/2, (y_hdr + y_hdr_bot)/2, lbl,
                fontsize=8.5, fontweight="bold",
                ha="center", va="center", color="#333")

    # ---- Data rows ----
    for i, r in enumerate(rows):
        y_top = y_top_of(i)
        y_bot = y_top - row_h_in / fig_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y_bot), 1, y_top - y_bot,
                                   facecolor="#fafafa", edgecolor="none"))

        # Fixed cells: pid / name / type
        for j_name, val in (("pid", r["pid"]), ("name", r["name"]), ("type", r["type"])):
            j = next(idx for idx, (l, _) in enumerate(cols) if l == j_name)
            w = cols[j][1]
            cx, cy = xs[j] + w/2, (y_top + y_bot) / 2
            ha = "left" if j_name == "name" else "center"
            tx = xs[j] + 0.006 if ha == "left" else cx
            style = "italic" if j_name in ("name", "type") else "normal"
            ax.text(tx, cy, str(val), fontsize=7.5,
                    ha=ha, va="center", color="#222", style=style)

        # Per-run cells
        for ri, prun in enumerate(r["per_run"]):
            col_status = 3 + ri * 3
            col_rate   = col_status + 1
            col_sp     = col_status + 2

            # status pill
            w = cols[col_status][1]
            pad_x = 0.012; pad_y = 0.003
            ax.add_patch(Rectangle((xs[col_status]+pad_x, y_bot+pad_y),
                                   w-2*pad_x, (y_top-y_bot)-2*pad_y,
                                   facecolor=prun["col"], edgecolor="none"))
            ax.text(xs[col_status]+w/2, (y_top+y_bot)/2, prun["lbl"],
                    fontsize=7.5, fontweight="bold",
                    ha="center", va="center", color="#222")

            # rate
            w = cols[col_rate][1]
            ax.text(xs[col_rate]+w/2, (y_top+y_bot)/2, prun["rate"],
                    fontsize=7.5, ha="center", va="center", color="#222")

            # speedup
            w = cols[col_sp][1]
            color = "#aaa" if prun["sp"] == "—" else "#222"
            ax.text(xs[col_sp]+w/2, (y_top+y_bot)/2, prun["sp"],
                    fontsize=7.5, ha="center", va="center", color=color)

        # Trailing cells (steps, elapsed)
        for j_name, val in (("steps", r["steps"]), ("elapsed", r["elapsed"])):
            j_iter = [idx for idx, (l, _) in enumerate(cols) if l == j_name]
            if not j_iter:
                continue
            j = j_iter[0]
            w = cols[j][1]
            ax.text(xs[j]+w/2, (y_top+y_bot)/2, str(val),
                    fontsize=7.5, ha="center", va="center", color="#666")

    # outer border
    y_table_top = y_top_of(0)
    y_table_bot = y_top_of(n)
    ax.add_patch(Rectangle((0, y_table_bot), 1, y_table_top - y_table_bot,
                           fill=False, edgecolor="#cccccc", linewidth=0.8))

    # ---- Summary section ----
    y_sum_top = y_table_bot - 0.025
    ax.text(0.005, y_sum_top, "Summary", fontsize=10, fontweight="bold", va="top")

    line_y = y_sum_top - 0.030
    line_h = 0.026

    # Per-category × per-run pass counts
    cats = {"pareval": [], "hecbench": [], "original": []}
    for r in rows:
        cats[category_for(r["pid"])].append(r)

    ax.text(0.020, line_y, "category", fontsize=8.5,
            fontweight="bold", va="top", color="#333")
    for ri, run in enumerate(runs):
        ax.text(0.140 + 0.15*ri, line_y, run["label"][:18],
                fontsize=8.5, fontweight="bold", va="top", color="#333")
    line_y -= line_h

    for cat in ("pareval", "hecbench", "original"):
        rs = cats[cat]
        ax.text(0.020, line_y, cat, fontsize=9, va="top")
        for ri in range(n_runs):
            p = sum(1 for r in rs if r["per_run"][ri]["lbl"] == "PASS")
            ax.text(0.140 + 0.15*ri, line_y, f"{p}/{len(rs)}",
                    fontsize=9, va="top")
        line_y -= line_h

    # Total per run
    line_y -= 0.005
    ax.text(0.020, line_y, "TOTAL", fontsize=10, fontweight="bold", va="top")
    for ri in range(n_runs):
        p = sum(1 for r in rows if r["per_run"][ri]["lbl"] == "PASS")
        ax.text(0.140 + 0.15*ri, line_y,
                f"{p}/{n} ({p/n*100:.0f}%)",
                fontsize=9, va="top",
                fontweight="bold" if ri == 0 else "normal")
    line_y -= line_h

    # Speedup geomeans per run
    line_y -= 0.005
    ax.text(0.020, line_y, "geomean speedup",
            fontsize=9, fontweight="bold", va="top", color="#333")
    line_y -= line_h
    for ri, run in enumerate(runs):
        e2e_xs = []; k_xs = []
        for ent in run["eval"].values():
            sp = ent.get("speedup")
            if not sp: continue
            if sp.get("speedup_e2e")    is not None: e2e_xs.append(sp["speedup_e2e"])
            if sp.get("speedup_kernel") is not None: k_xs.append(sp["speedup_kernel"])
        e2e_g = geomean(e2e_xs); k_g = geomean(k_xs)
        ax.text(0.020, line_y, run["label"], fontsize=9, va="top")
        ax.text(0.200, line_y,
                f"e2e={_num(e2e_g)}    kernel={_num(k_g)}    "
                f"(n={len(k_xs)})",
                fontsize=9, va="top", color="#222")
        line_y -= line_h

    fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=160)
    print(f"wrote {out_path}  ({fig_w:.1f}x{fig_h:.1f}in)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+",
                    help="One or more run directories (each must contain "
                         "eval_results.json)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Custom label per run (default: dir basename)")
    ap.add_argument("--title", default="ParallelMind Eval Suite",
                    help="Figure title")
    ap.add_argument("--out", default=None,
                    help="Output PNG (default: <first-run>/comparison.png)")
    ap.add_argument("--benchmarks", default=str(DEFAULT_BENCHMARKS))
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        ap.error("--labels count must match --runs count")

    bench = json.loads(Path(args.benchmarks).read_text())
    if "problems" not in bench:
        raise SystemExit(f"{args.benchmarks}: no 'problems' key")

    runs = [load_run(Path(d), label=(args.labels[i] if args.labels else None))
            for i, d in enumerate(args.runs)]

    out_path = Path(args.out) if args.out else (Path(args.runs[0]) / "comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render(runs, bench, out_path, args.title)


if __name__ == "__main__":
    main()
