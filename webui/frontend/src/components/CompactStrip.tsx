import { Link } from "react-router-dom";
import type { EvalEntry, RunSummary } from "../api/client";
import { classify, type StatusKind } from "./StatusPill";
import { fmtSpeedup, geomean } from "./SpeedupCell";

type EvalByRun = Record<string, Record<string, EvalEntry>>;

const CELL_BG: Record<StatusKind, string> = {
  PASS:    "bg-pass",
  PARTIAL: "bg-partial",
  BUILD:   "bg-build",
  FAIL:    "bg-fail",
  NONE:    "bg-slate-200 dark:bg-slate-700",
};

function shortRunLabel(name: string) {
  return name.replace(/^.*?[\/=]/, "");
}

function fmtSp(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 100) return `${n.toFixed(0)}×`;
  if (n >= 10)  return `${n.toFixed(1)}×`;
  return `${n.toFixed(2)}×`;
}

export default function CompactStrip({
  runs,
  pids,
  evals,
}: {
  runs: RunSummary[];
  pids: string[];
  evals: EvalByRun;
}) {
  if (runs.length === 0) {
    return (
      <div className="text-sm text-slate-500 py-6">
        Select one or more runs to compare.
      </div>
    );
  }

  // Pre-compute every run's cells + stats once.
  const rows = runs.map((r) => {
    let pass = 0, partial = 0, fail = 0, build = 0, none = 0;
    const kern: number[] = [];
    const cells = pids.map((pid) => {
      const ent = evals[r.name]?.[pid];
      const c = classify(ent);
      if (c.kind === "PASS")    pass++;
      else if (c.kind === "PARTIAL") partial++;
      else if (c.kind === "FAIL")    fail++;
      else if (c.kind === "BUILD")   build++;
      else none++;
      const sp = ent?.speedup?.speedup_kernel;
      if (sp != null && c.kind === "PASS") kern.push(sp);
      return { pid, kind: c.kind, rate: c.rate, sp };
    });
    return {
      run: r,
      cells,
      stats: { pass, partial, fail, build, none, kern: geomean(kern), n: kern.length },
    };
  });

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-md p-4 bg-white dark:bg-slate-900 space-y-3">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Pass status × {pids.length} problems
      </div>

      {/* Legend, aligned to the strip column so colors sit directly above cells. */}
      <div
        className="flex gap-3 text-xs text-slate-500"
        style={{ marginLeft: 260 + 12 /* run-label width + gap */ }}
      >
        <Legend kind="PASS" label="pass" />
        <Legend kind="PARTIAL" label="partial" />
        <Legend kind="FAIL" label="fail" />
        <Legend kind="BUILD" label="build" />
        <Legend kind="NONE" label="no-sub" />
      </div>

      <div className="space-y-1.5">
        {rows.map(({ run, cells, stats }) => (
          <div key={run.name} className="flex items-center gap-3">
            {/* Run label */}
            <div
              className="font-mono text-xs text-slate-700 dark:text-slate-200 truncate text-right"
              style={{ width: 260 }}
              title={run.model_name ?? run.name}
            >
              {shortRunLabel(run.name)}
            </div>

            {/* 30-cell strip */}
            <div className="flex gap-[2px]">
              {cells.map(({ pid, kind, rate, sp }) => {
                const title = `${pid} · ${kind}${rate ? ` ${rate}` : ""}${
                  sp != null ? ` · kern ${fmtSp(sp)}` : ""
                }`;
                return (
                  <Link
                    key={pid}
                    to={`/trace?run=${encodeURIComponent(run.name)}&pid=${pid}`}
                    title={title}
                    className={`block w-3.5 h-5 ${CELL_BG[kind]} rounded-[2px]
                                hover:ring-2 hover:ring-indigo-400 hover:z-10`}
                  />
                );
              })}
            </div>

            {/* Stats */}
            <div className="font-mono text-xs text-slate-600 dark:text-slate-300 whitespace-nowrap">
              <span className="font-semibold text-slate-900 dark:text-slate-100">
                {stats.pass}
              </span>
              <span className="opacity-60">/{pids.length}</span>
              {stats.partial > 0 && (
                <span className="ml-2 text-amber-600 dark:text-amber-400">
                  +{stats.partial} part
                </span>
              )}
              {stats.kern != null && (
                <span className="ml-3 opacity-80">
                  kern {fmtSpeedup(stats.kern)}
                  <span className="opacity-50"> (n={stats.n})</span>
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* X-axis pid ticks */}
      <div
        className="flex text-[10px] font-mono text-slate-400"
        style={{ marginLeft: 260 + 12 /* gap */ }}
      >
        {pids.map((pid, i) => {
          const n = parseInt(pid.slice(1), 10);
          const show = n === 1 || n % 5 === 0;
          return (
            <div
              key={pid}
              className="flex justify-start"
              style={{ width: 14 + 2 /* cell + gap */ }}
            >
              {show && i > 0 && <span className="-ml-1">{pid}</span>}
              {show && i === 0 && <span>{pid}</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Legend({ kind, label }: { kind: StatusKind; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className={`inline-block w-3 h-3 rounded-[2px] ${CELL_BG[kind]}`} />
      {label}
    </span>
  );
}
