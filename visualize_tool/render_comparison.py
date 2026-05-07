#!/usr/bin/env python3
"""Render the bare-vs-agent comparison table as a PNG.

Inputs:
  - eval/benchmarks.json
  - eval/submissions.json                           (per-candidate validation)
  - parallelmind_harness/runs/legacy/agent_v1/...   (agent steps + elapsed)
  - parallelmind_harness/runs/legacy/timing.json    (end-to-end + kernel-only)

Output:
  - eval/comparison_table.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EVAL = Path(__file__).resolve().parent
HARNESS_RUNS = EVAL.parent / "runs"

BARE_TAG  = "qwen3-coder_v2"
AGENT_TAG = "qwen3-coder_agent_v1"

C_PASS    = "#a3d8a3"
C_FAIL    = "#f0928b"
C_BUILD   = "#f0b870"
C_PARTIAL = "#f5d870"
C_NA      = "#dddddd"
C_ROW_HL  = "#fff3da"


def category_for(pid: str) -> str:
    n = int(pid[1:])
    if n <= 10:  return "pareval"
    if n <= 20: return "hecbench"
    return "original"


def short_name(name: str) -> str:
    for pre, repl in (("pareval_", "p_"),
                      ("hecbench_", "h_"),
                      ("original_", "o_")):
        if name.startswith(pre):
            return repl + name[len(pre):]
    return name


def status_label(cand: dict | None) -> tuple[str, str]:
    if not cand:
        return ("—", C_NA)
    if not cand.get("validation"):
        return ("—", C_NA)
    summ = cand["validation"]["summary"]
    total = summ["total"]
    if total <= 1:
        return ("BUILD", C_BUILD)
    passed = summ["pass_byte"] + summ["pass_checker"] + summ["pass_llm"]
    if passed == total:
        return ("PASS", C_PASS)
    if passed == 0:
        return ("FAIL", C_FAIL)
    return ("PARTIAL", C_PARTIAL)


def passed_total(cand: dict | None) -> tuple[int, int] | None:
    if not cand or not cand.get("validation"):
        return None
    s = cand["validation"]["summary"]
    return (s["pass_byte"] + s["pass_checker"] + s["pass_llm"], s["total"])


def fmt_speedup(timing: dict, pid: str, tag: str) -> str:
    e = timing.get(pid, {}).get(tag)
    if not e:
        return "—"
    e2e = e.get("speedup_e2e")
    k   = e.get("speedup_kernel")

    def num(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        if x >= 100: return f"{x:.0f}×"
        if x >= 10:  return f"{x:.1f}×"
        if x >= 0.1: return f"{x:.2f}×"
        if x > 0:    return f"{x:.3f}×"
        return f"{x:.2f}×"

    return f"{num(e2e)} / {num(k)}"


def main() -> None:
    bench  = json.loads((EVAL / "benchmarks.json").read_text())
    subs   = json.loads((EVAL / "submissions.json").read_text())
    agent  = {e["id"]: e for e in json.loads(
        (HARNESS_RUNS / "legacy" / "agent_v1" / "agent_output.json").read_text())}
    timing_path = HARNESS_RUNS / "legacy" / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    pids = sorted(bench["problems"].keys())

    # --- Build rows ---
    rows = []
    for pid in pids:
        prob = bench["problems"][pid]
        cands = (subs["submissions"].get(pid) or {}).get("candidates") or {}
        bare = cands.get(BARE_TAG)
        agt  = cands.get(AGENT_TAG)

        bare_lbl, bare_color = status_label(bare)
        agt_lbl,  agt_color  = status_label(agt)

        bare_pt = passed_total(bare)
        agt_pt  = passed_total(agt)

        det = "byte" if prob.get("byte_deterministic") else "checker"
        a = agent.get(pid, {})
        steps = a.get("agent_steps", "")
        elapsed_s = a.get("agent_elapsed_s")
        elapsed = ""
        if isinstance(elapsed_s, (int, float)):
            m, s = divmod(int(elapsed_s), 60)
            elapsed = f"{m}m{s:02d}s"

        rows.append({
            "pid": pid,
            "name": short_name(prob["name"]),
            "type": det,
            "bare_lbl": bare_lbl, "bare_c": bare_color,
            "bare_rate": f"{bare_pt[0]}/{bare_pt[1]}" if bare_pt else "",
            "bare_speedup": fmt_speedup(timing, pid, BARE_TAG),
            "agt_lbl": agt_lbl,   "agt_c":  agt_color,
            "agt_rate":  f"{agt_pt[0]}/{agt_pt[1]}" if agt_pt else "",
            "agt_speedup": fmt_speedup(timing, pid, AGENT_TAG),
            "steps": steps,
            "elapsed": elapsed,
            "regress": (bare_lbl == "PASS" and agt_lbl != "PASS"),
        })

    # --- Layout ---
    cols = [
        ("pid",          0.045),
        ("name",         0.180),
        ("type",         0.060),
        ("BARE",         0.075),
        ("rate",         0.050),
        ("e2e / kern",   0.085),
        ("AGENT",        0.075),
        ("rate",         0.050),
        ("e2e / kern",   0.085),
        ("steps",        0.050),
        ("elapsed",      0.060),
    ]
    field_keys = ["pid","name","type","bare_lbl","bare_rate","bare_speedup",
                  "agt_lbl","agt_rate","agt_speedup","steps","elapsed"]
    color_keys = {3: "bare_c", 6: "agt_c"}  # which columns get colored cells

    # Compute x positions
    xs = []
    x = 0.0
    for _, w in cols:
        xs.append(x)
        x += w
    table_w = x  # total width fraction (≈1.0)

    # rows + summary section
    n = len(rows)
    fig_w = 9.5
    row_h_in = 0.21
    header_h = 0.50
    title_h = 0.55
    summary_h = 1.55
    fig_h = title_h + header_h + n * row_h_in + summary_h + 0.25

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    ax = fig.add_axes([0.012, 0.015, 0.976, 0.97])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # convert vertical space to 0..1 fractions
    def y_top_of(i_row):
        # i_row=0 is top row of table
        return 1 - (title_h + header_h + i_row * row_h_in) / fig_h

    # --- Title ---
    ax.text(0.0, 1 - 0.20/fig_h,
            "ParallelMind Eval Suite — Qwen3-Coder × 30 problems",
            fontsize=14, fontweight="bold", va="top")
    ax.text(0.0, 1 - 0.42/fig_h,
            "bare (single-shot) vs agent (tool-calling loop)",
            fontsize=9, color="#666", va="top")

    # Legend (top right)
    legend_items = [("PASS", C_PASS), ("FAIL", C_FAIL),
                    ("BUILD", C_BUILD), ("PARTIAL", C_PARTIAL)]
    lx = 0.66
    for label, col in legend_items:
        ax.add_patch(Rectangle((lx, 1 - 0.30/fig_h), 0.035, 0.018,
                               facecolor=col, edgecolor="none",
                               transform=ax.transAxes))
        ax.text(lx + 0.04, 1 - 0.295/fig_h, label,
                fontsize=8, va="center")
        lx += 0.085

    # --- Header row ---
    y_hdr = 1 - (title_h + 0.05) / fig_h
    y_hdr_bot = 1 - (title_h + header_h) / fig_h
    ax.add_patch(Rectangle((0, y_hdr_bot), 1, y_hdr - y_hdr_bot,
                           facecolor="#f4f4f4", edgecolor="none"))
    for i, (label, w) in enumerate(cols):
        ax.text(xs[i] + w/2, (y_hdr + y_hdr_bot)/2, label,
                fontsize=8.5, fontweight="bold",
                ha="center", va="center", color="#333")

    # Speedup column subheading
    ax.text(xs[5] + cols[5][1]/2, y_hdr_bot - 0.005, "(BARE)",
            fontsize=6.5, color="#888", ha="center", va="top")
    ax.text(xs[8] + cols[8][1]/2, y_hdr_bot - 0.005, "(AGENT)",
            fontsize=6.5, color="#888", ha="center", va="top")

    # --- Data rows ---
    for i, r in enumerate(rows):
        y_top = y_top_of(i)
        y_bot = y_top - row_h_in / fig_h
        if r["regress"]:
            ax.add_patch(Rectangle((0.002, y_bot), 0.996, y_top - y_bot,
                                   facecolor=C_ROW_HL,
                                   edgecolor="#e0a04a",
                                   linewidth=0.9))
        # alternating subtle stripe
        elif i % 2 == 0:
            ax.add_patch(Rectangle((0, y_bot), 1, y_top - y_bot,
                                   facecolor="#fafafa", edgecolor="none"))

        for j, (_, w) in enumerate(cols):
            key = field_keys[j]
            val = r[key]
            cx  = xs[j] + w/2
            cy  = (y_top + y_bot) / 2
            if j in color_keys:
                # colored status pill
                col = r[color_keys[j]]
                pad_x = 0.012
                pad_y = 0.003
                ax.add_patch(Rectangle((xs[j]+pad_x, y_bot+pad_y),
                                       w-2*pad_x, (y_top-y_bot)-2*pad_y,
                                       facecolor=col, edgecolor="none"))
                ax.text(cx, cy, val, fontsize=7.5, fontweight="bold",
                        ha="center", va="center", color="#222")
            else:
                ha = "left" if key == "name" else "center"
                tx = xs[j] + 0.006 if ha == "left" else cx
                color = "#222"
                style = "italic" if key in ("name", "type") else "normal"
                if key in ("bare_speedup","agt_speedup") and val == "—":
                    color = "#aaa"
                ax.text(tx, cy, str(val), fontsize=7.5,
                        ha=ha, va="center", color=color, style=style)

    # outer border
    y_table_top = y_top_of(0)
    y_table_bot = y_top_of(n)
    ax.add_patch(Rectangle((0, y_table_bot), 1, y_table_top - y_table_bot,
                           fill=False, edgecolor="#cccccc", linewidth=0.8))

    # --- Summary section ---
    cats = {"pareval": [], "hecbench": [], "original": []}
    for r in rows:
        cats[category_for(r["pid"])].append(r)

    bare_total = sum(1 for r in rows if r["bare_lbl"] == "PASS")
    agt_total  = sum(1 for r in rows if r["agt_lbl"]  == "PASS")
    delta = agt_total - bare_total
    delta_color = "#1c8a3e" if delta > 0 else ("#b73838" if delta < 0 else "#666")

    elapsed_total_s = sum(
        agent[r["pid"]].get("agent_elapsed_s") or 0 for r in rows
    )
    em, es = divmod(int(elapsed_total_s), 60)
    eh, em = divmod(em, 60)
    elapsed_total_str = f"{eh}h{em:02d}m{es:02d}s" if eh else f"{em}m{es:02d}s"

    avg_steps_total = sum(
        agent[r["pid"]].get("agent_steps") or 0 for r in rows
    ) / max(n, 1)

    y_sum_top = y_table_bot - 0.025
    ax.text(0.005, y_sum_top, "Summary", fontsize=10, fontweight="bold",
            va="top")

    line_y = y_sum_top - 0.030
    line_h = 0.026
    for cat in ("pareval", "hecbench", "original"):
        rs = cats[cat]
        b_pass = sum(1 for r in rs if r["bare_lbl"] == "PASS")
        a_pass = sum(1 for r in rs if r["agt_lbl"]  == "PASS")
        d = a_pass - b_pass
        dc = "#1c8a3e" if d > 0 else ("#b73838" if d < 0 else "#666")
        avg_s = sum(agent[r["pid"]].get("agent_steps") or 0 for r in rs) / max(len(rs),1)
        ax.text(0.020, line_y, cat,                fontsize=9, va="top")
        ax.text(0.140, line_y, f"bare={b_pass}/{len(rs)}",   fontsize=9, va="top")
        ax.text(0.270, line_y, f"agent={a_pass}/{len(rs)}",  fontsize=9, va="top")
        ax.text(0.410, line_y, f"Δ {d:+d}", fontsize=9, va="top", color=dc)
        ax.text(0.520, line_y, f"avg steps {avg_s:.1f}", fontsize=9, va="top")
        line_y -= line_h

    # --- speedup summary (geomean) ---
    def geomean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))
              and x is not None and not math.isnan(x) and x > 0]
        if not xs:
            return None
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    bare_e2e = []; bare_k = []; agt_e2e = []; agt_k = []
    for pid, t in timing.items():
        be = t.get(BARE_TAG)
        ae = t.get(AGENT_TAG)
        if be:
            if be.get("speedup_e2e")    is not None: bare_e2e.append(be["speedup_e2e"])
            if be.get("speedup_kernel") is not None: bare_k.append(be["speedup_kernel"])
        if ae:
            if ae.get("speedup_e2e")    is not None: agt_e2e.append(ae["speedup_e2e"])
            if ae.get("speedup_kernel") is not None: agt_k.append(ae["speedup_kernel"])

    def fmt_g(v):
        if v is None: return "—"
        if v >= 100: return f"{v:.0f}×"
        if v >= 10:  return f"{v:.1f}×"
        return f"{v:.2f}×"

    line_y -= 0.005
    ax.text(0.020, line_y, "geomean speedup",
            fontsize=9, fontweight="bold", va="top", color="#333")
    line_y -= line_h
    ax.text(0.020, line_y, "bare",  fontsize=9, va="top")
    ax.text(0.140, line_y, f"e2e={fmt_g(geomean(bare_e2e))}",
            fontsize=9, va="top")
    ax.text(0.270, line_y, f"kernel={fmt_g(geomean(bare_k))}",
            fontsize=9, va="top")
    line_y -= line_h
    ax.text(0.020, line_y, "agent", fontsize=9, va="top")
    ax.text(0.140, line_y, f"e2e={fmt_g(geomean(agt_e2e))}",
            fontsize=9, va="top")
    ax.text(0.270, line_y, f"kernel={fmt_g(geomean(agt_k))}",
            fontsize=9, va="top")
    line_y -= line_h

    # bottom: TOTAL row
    line_y -= 0.012
    hit_max = sum(1 for r in rows
                  if (agent[r["pid"]].get("agent_steps") or 0) >= 30)
    ax.text(0.020, line_y, "TOTAL", fontsize=10, fontweight="bold", va="top")
    ax.text(0.140, line_y,
            f"bare={bare_total}/{n} ({bare_total/n*100:.0f}%)",
            fontsize=9, va="top")
    ax.text(0.300, line_y,
            f"agent={agt_total}/{n} ({agt_total/n*100:.0f}%)",
            fontsize=9, va="top")
    ax.text(0.470, line_y, f"Δ {delta:+d}",
            fontsize=9, va="top", color=delta_color)
    ax.text(0.560, line_y, f"avg steps {avg_steps_total:.1f}",
            fontsize=9, va="top")
    if hit_max:
        ax.text(0.020, line_y - line_h,
                f"({hit_max} hit max_steps=30)",
                fontsize=8, color="#b73838", va="top", style="italic")

    out = EVAL / "comparison_table.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=160)
    print(f"wrote {out}  ({fig_w:.1f}x{fig_h:.1f}in)")


if __name__ == "__main__":
    main()
