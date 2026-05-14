import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  type AgentOutputEntry,
  type Benchmarks,
  type EvalEntry,
  type RunSummary,
} from "../api/client";
import StatusPill, { classify } from "../components/StatusPill";
import SpeedupCell, { fmtSpeedup, geomean } from "../components/SpeedupCell";
import RunPicker from "../components/RunPicker";
import CompactStrip from "../components/CompactStrip";

type ViewMode = "table" | "compact";

type EvalByRun  = Record<string, Record<string, EvalEntry>>;
type AgentByRun = Record<string, Record<string, AgentOutputEntry>>;

function fmtElapsed(s: number | undefined): string {
  if (s === undefined || s === null) return "—";
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s - m * 60);
  return `${m}m${r.toString().padStart(2, "0")}s`;
}

function shortName(name: string) {
  return name
    .replace(/^pareval_/, "p_")
    .replace(/^hecbench_/, "h_")
    .replace(/^original_/, "o_");
}

function categoryFor(pid: string): "pareval" | "hecbench" | "original" {
  const n = parseInt(pid.slice(1), 10);
  if (n <= 10) return "pareval";
  if (n <= 20) return "hecbench";
  return "original";
}

function shortLabel(name: string) {
  // For headers in the table; trim long model names.
  return name.replace(/^.*?[\/=]/, "").slice(0, 28);
}

export default function Analyzer() {
  const [runs, setRuns]       = useState<RunSummary[]>([]);
  const [bench, setBench]     = useState<Benchmarks | null>(null);
  const [selected, setSel]    = useState<Set<string>>(new Set());
  const [evals, setEvals]     = useState<EvalByRun>({});
  const [agents, setAgents]   = useState<AgentByRun>({});
  const [openPid, setOpenPid] = useState<string | null>(null);
  const [err, setErr]         = useState<string | null>(null);
  const [view, setView]       = useState<ViewMode>("table");

  // Load runs + benchmarks once.
  useEffect(() => {
    Promise.all([api.runs(), api.benchmarks()])
      .then(([rs, bk]) => {
        setRuns(rs);
        setBench(bk);
        // Default: select all runs with eval_results.
        setSel(new Set(rs.filter((r) => r.has_eval).map((r) => r.name)));
      })
      .catch((e) => setErr(String(e)));
  }, []);

  // Lazy-fetch eval_results for newly-selected runs that actually have
  // one. Use allSettled so a missing file doesn't blow up the page —
  // those runs just render with empty (NONE) cells.
  useEffect(() => {
    const hasEval = (n: string) => runs.find((r) => r.name === n)?.has_eval;
    const missing = [...selected].filter((n) => !evals[n] && hasEval(n));
    if (missing.length === 0) return;
    Promise.allSettled(
      missing.map((n) => api.evalResults(n).then(es => [n, es] as const)),
    ).then((settled) => {
      setEvals((prev) => {
        const next = { ...prev };
        for (const r of settled) {
          if (r.status === "fulfilled") {
            const [n, es] = r.value;
            next[n] = Object.fromEntries(es.map((e) => [e.id, e]));
          } else {
            console.warn("eval_results fetch failed:", r.reason);
          }
        }
        return next;
      });
    });
  }, [selected, runs]);

  // Lazy-fetch agent_output (for steps + elapsed in the drawer).
  useEffect(() => {
    const missing = [...selected].filter((n) => !agents[n]);
    if (missing.length === 0) return;
    Promise.allSettled(
      missing.map((n) => api.agentOutput(n).then(es => [n, es] as const)),
    ).then((settled) => {
      setAgents((prev) => {
        const next = { ...prev };
        for (const r of settled) {
          if (r.status === "fulfilled") {
            const [n, es] = r.value;
            next[n] = Object.fromEntries(es.map((e) => [e.id, e]));
          }
        }
        return next;
      });
    });
  }, [selected]);

  const selectedRuns = useMemo(
    () => runs.filter((r) => selected.has(r.name)),
    [runs, selected],
  );

  const pids = useMemo(() => (bench ? Object.keys(bench.problems).sort() : []), [bench]);

  // ---- Summary numbers (per category × per run, geomean speedup) ----
  const summary = useMemo(() => {
    type Row = { pareval: number; hecbench: number; original: number; total: number };
    const perRun: Record<string, Row & { e2e: number[]; kern: number[] }> = {};
    for (const r of selectedRuns) {
      perRun[r.name] = { pareval: 0, hecbench: 0, original: 0, total: 0, e2e: [], kern: [] };
    }
    for (const pid of pids) {
      for (const r of selectedRuns) {
        const ent = evals[r.name]?.[pid];
        const c = classify(ent);
        if (c.kind === "PASS") {
          perRun[r.name][categoryFor(pid)] += 1;
          perRun[r.name].total += 1;
        }
        const sp = ent?.speedup;
        if (sp?.speedup_e2e != null) perRun[r.name].e2e.push(sp.speedup_e2e);
        if (sp?.speedup_kernel != null) perRun[r.name].kern.push(sp.speedup_kernel);
      }
    }
    return perRun;
  }, [evals, selectedRuns, pids]);

  if (err) return <div className="p-6 text-red-600 text-sm font-mono">{err}</div>;
  if (!bench) return <div className="p-6 text-slate-500">Loading…</div>;

  const detailProb = openPid ? bench.problems[openPid] : null;

  return (
    <div className="max-w-screen-2xl mx-auto p-6 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold mb-2">Analyzer</h2>
          <p className="text-sm text-slate-500">
            Cross-run validation + speedup table. Click a row to expand the
            problem spec; click a PASS/FAIL pill to jump to the agent trace.
          </p>
        </div>
        <div className="flex border border-slate-300 dark:border-slate-600 rounded-md overflow-hidden text-xs font-mono">
          {(["table", "compact"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setView(m)}
              className={`px-3 py-1.5 ${
                view === m
                  ? "bg-slate-800 text-white dark:bg-slate-200 dark:text-slate-900"
                  : "bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="border-b border-slate-200 pb-3">
        <div className="text-xs uppercase text-slate-500 mb-2">runs</div>
        <RunPicker runs={runs} selected={selected} onChange={setSel} />
      </div>

      {view === "compact" && (
        <CompactStrip runs={selectedRuns} pids={pids} evals={evals} />
      )}

      {view === "table" && (<>
      {/* Main table */}
      <div className="overflow-x-auto border border-slate-200 dark:border-slate-700 rounded-md">
        <table className="text-sm w-full">
          <thead className="bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200">
            <tr>
              <th className="px-3 py-2 text-left font-semibold">pid</th>
              <th className="px-3 py-2 text-left font-semibold">name</th>
              <th className="px-3 py-2 text-left font-semibold">type</th>
              {selectedRuns.map((r) => (
                <th
                  key={r.name}
                  colSpan={3}
                  className="px-3 py-1.5 text-center font-semibold border-l border-slate-300 dark:border-slate-600"
                  title={r.model_name ?? ""}
                >
                  <div>{shortLabel(r.name)}</div>
                  {!r.has_eval && (
                    <div className="text-[10px] font-normal text-amber-600 dark:text-amber-400">
                      no eval_results
                    </div>
                  )}
                </th>
              ))}
            </tr>
            <tr className="text-[10px] uppercase text-slate-400">
              <th></th>
              <th></th>
              <th></th>
              {selectedRuns.map((r) => (
                <>
                  <th key={r.name + "-s"} className="px-3 py-1 font-normal border-l border-slate-300 dark:border-slate-600">status</th>
                  <th key={r.name + "-r"} className="px-3 py-1 font-normal">rate</th>
                  <th key={r.name + "-sp"} className="px-3 py-1 font-normal">e2e / kern</th>
                </>
              ))}
            </tr>
          </thead>
          <tbody>
            {pids.map((pid, idx) => {
              const prob = bench.problems[pid];
              const isOpen = openPid === pid;
              const type = prob.byte_deterministic === false ? "checker" : "byte";
              return (
                <>
                  <tr
                    key={pid}
                    className={`border-t border-slate-100 dark:border-slate-800 ${
                      idx % 2 === 0 ? "bg-white dark:bg-slate-900" : "bg-slate-50/50 dark:bg-slate-900/50"
                    } ${isOpen ? "ring-2 ring-indigo-400 ring-inset" : ""}`}
                    onClick={() => setOpenPid(isOpen ? null : pid)}
                  >
                    <td className="px-3 py-1.5 font-mono text-xs cursor-pointer">{pid}</td>
                    <td className="px-3 py-1.5 italic font-mono text-xs cursor-pointer">{shortName(prob.name)}</td>
                    <td className="px-3 py-1.5 italic text-xs cursor-pointer">{type}</td>
                    {selectedRuns.map((r) => {
                      const ent = evals[r.name]?.[pid];
                      const c = classify(ent);
                      return (
                        <>
                          <td key={r.name + "-s"} className="px-3 py-1 text-center border-l border-slate-100 dark:border-slate-800" onClick={(e) => e.stopPropagation()}>
                            <Link to={`/trace?run=${encodeURIComponent(r.name)}&pid=${pid}`}>
                              <StatusPill kind={c.kind} onClick={() => {}} />
                            </Link>
                          </td>
                          <td key={r.name + "-r"} className="px-3 py-1 text-center font-mono text-xs text-slate-600 cursor-pointer">{c.rate}</td>
                          <td key={r.name + "-sp"} className="px-3 py-1 text-center cursor-pointer">
                            <SpeedupCell entry={ent} />
                          </td>
                        </>
                      );
                    })}
                  </tr>
                  {isOpen && (
                    <tr className="bg-indigo-50/40 dark:bg-indigo-900/20 border-t border-indigo-200">
                      <td colSpan={3 + selectedRuns.length * 3} className="px-6 py-4">
                        <ProblemDrawer pid={pid} runs={selectedRuns} evals={evals} agents={agents} description={prob.description} />
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      <div className="border-t border-slate-200 dark:border-slate-700 pt-4">
        <h3 className="text-base font-semibold mb-3">Summary</h3>
        <div className="overflow-x-auto">
          <table className="text-sm">
            <thead className="text-slate-500">
              <tr>
                <th className="text-left px-3 py-1">category</th>
                {selectedRuns.map((r) => (
                  <th key={r.name} className="text-left px-4 py-1 font-mono text-xs">{r.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(["pareval", "hecbench", "original"] as const).map((cat) => (
                <tr key={cat} className="border-t border-slate-100">
                  <td className="px-3 py-1 italic">{cat}</td>
                  {selectedRuns.map((r) => (
                    <td key={r.name} className="px-4 py-1 font-mono text-xs">
                      {summary[r.name]?.[cat] ?? 0}/10
                    </td>
                  ))}
                </tr>
              ))}
              <tr className="border-t-2 border-slate-300 font-semibold">
                <td className="px-3 py-1.5">TOTAL</td>
                {selectedRuns.map((r) => {
                  const s = summary[r.name];
                  return (
                    <td key={r.name} className="px-4 py-1.5 font-mono text-xs">
                      {s?.total ?? 0}/{pids.length} ({Math.round(((s?.total ?? 0) / pids.length) * 100)}%)
                    </td>
                  );
                })}
              </tr>
              <tr className="text-slate-500 text-xs">
                <td className="px-3 pt-3">geomean</td>
                {selectedRuns.map((r) => {
                  const s = summary[r.name];
                  const e2e = s ? geomean(s.e2e) : null;
                  const k = s ? geomean(s.kern) : null;
                  return (
                    <td key={r.name} className="px-4 pt-3 font-mono">
                      e2e={fmtSpeedup(e2e)} kern={fmtSpeedup(k)}{" "}
                      <span className="opacity-60">(n={s?.kern.length ?? 0})</span>
                    </td>
                  );
                })}
              </tr>
              <tr className="text-slate-500 text-xs">
                <td className="px-3 pt-1">wall</td>
                {selectedRuns.map((r) => {
                  const ag = agents[r.name];
                  if (!ag) return <td key={r.name} className="px-4 pt-1 font-mono">—</td>;
                  const totalSec = Object.values(ag).reduce((s, e) => s + (e.elapsed_s ?? 0), 0);
                  const avgSteps = Object.values(ag).reduce((s, e) => s + (e.steps ?? 0), 0)
                                  / Math.max(1, Object.keys(ag).length);
                  return (
                    <td key={r.name} className="px-4 pt-1 font-mono">
                      Σ {fmtElapsed(totalSec)}{" "}
                      <span className="opacity-60">· avg {avgSteps.toFixed(1)} steps</span>
                    </td>
                  );
                })}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {detailProb && (
        <p className="text-xs text-slate-500">
          Tip: pass-pill click goes to /trace. Future phase: render-PNG export.
        </p>
      )}
      </>)}
    </div>
  );
}

function ProblemDrawer({
  pid,
  runs,
  evals,
  agents,
  description,
}: {
  pid: string;
  runs: RunSummary[];
  evals: EvalByRun;
  agents: AgentByRun;
  description: string;
}) {
  return (
    <div className="grid grid-cols-[1fr_2fr] gap-6">
      <div>
        <div className="text-xs uppercase text-slate-500 mb-1">description</div>
        <pre className="whitespace-pre-wrap text-xs font-mono leading-relaxed bg-slate-50 dark:bg-slate-800/60 p-3 rounded border border-slate-200 dark:border-slate-700">
          {description}
        </pre>
      </div>
      <div className="space-y-4">
        {runs.map((r) => {
          const ent = evals[r.name]?.[pid];
          const ag  = agents[r.name]?.[pid];
          const c = classify(ent);
          return (
            <div key={r.name} className="border border-slate-200 dark:border-slate-700 rounded p-3">
              <div className="flex items-center gap-3 mb-2">
                <span className="font-mono text-sm">{r.name}</span>
                <StatusPill kind={c.kind} />
                {c.rate && <span className="text-xs text-slate-500 font-mono">{c.rate}</span>}
                {ag && (
                  <span className="ml-auto text-xs text-slate-500 font-mono">
                    {ag.steps} steps · {fmtElapsed(ag.elapsed_s)}
                  </span>
                )}
              </div>
              {ent?.validation && ent.validation.cases.length <= 12 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {ent.validation.cases.map((cs, i) => (
                    <span
                      key={i}
                      className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${
                        cs.status.startsWith("PASS")
                          ? "bg-pass/70" : "bg-fail/70"
                      }`}
                      title={cs.status + (cs.detail ? `: ${cs.detail}` : "")}
                    >
                      {cs.status.replace("PASS_", "").replace("FAIL", "F")}
                    </span>
                  ))}
                </div>
              )}
              {ent?.speedup && (
                <div className="text-xs font-mono text-slate-600 dark:text-slate-300">
                  <SpeedupCell entry={ent} />
                </div>
              )}
              {!ent?.submitted && (
                <div className="text-xs text-slate-500">not submitted</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
