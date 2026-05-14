import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  api,
  type AgentOutputEntry,
  type RunSummary,
  type ToolCallLog,
  type TraceMessage,
} from "../api/client";

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms - m * 60_000) / 1000);
  return `${m}m${s.toString().padStart(2, "0")}s`;
}

const ROLE_COLOR: Record<string, string> = {
  system:    "border-slate-300 bg-slate-50 dark:bg-slate-800/60",
  user:      "border-emerald-300 bg-emerald-50 dark:bg-emerald-900/20",
  assistant: "border-indigo-300 bg-indigo-50 dark:bg-indigo-900/20",
  tool:      "border-amber-300 bg-amber-50 dark:bg-amber-900/20",
};

const ROLE_LABEL: Record<string, string> = {
  system: "SYSTEM", user: "USER", assistant: "ASSISTANT", tool: "TOOL",
};

function tryParseJSON(s: string): unknown {
  try { return JSON.parse(s); } catch { return null; }
}

function PreviousAnalysisFold({ content }: { content: string }) {
  // [previous analysis] tag is harness-injected reasoning echo (loop.py
  // pre-pends reasoning when --reasoning); fold it so the actionable
  // assistant text reads cleanly.
  const m = content.match(/^(\[previous analysis\][\s\S]*?)(?:\n\n|\n*$)/);
  if (!m) return null;
  const ra = m[1];
  const rest = content.slice(m[0].length);
  return (
    <>
      <details className="mb-2">
        <summary className="cursor-pointer text-xs text-slate-500 select-none">
          ↳ reasoning ({ra.length} chars)
        </summary>
        <pre className="whitespace-pre-wrap text-xs font-mono mt-1 text-slate-500 border-l-2 border-slate-300 dark:border-slate-600 pl-3">
          {ra}
        </pre>
      </details>
      <pre className="whitespace-pre-wrap text-sm font-mono">{rest}</pre>
    </>
  );
}

function MessageContent({ content }: { content: string }) {
  if (!content) return <span className="text-slate-400 text-xs italic">(empty)</span>;
  if (content.startsWith("[previous analysis]")) {
    return <PreviousAnalysisFold content={content} />;
  }
  return <pre className="whitespace-pre-wrap text-sm font-mono">{content}</pre>;
}

function ToolCalls({ msg }: { msg: TraceMessage }) {
  if (!msg.tool_calls?.length) return null;
  return (
    <div className="mt-2 space-y-1">
      {msg.tool_calls.map((tc) => {
        const args = tryParseJSON(tc.function.arguments);
        return (
          <details key={tc.id} className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded">
            <summary className="cursor-pointer px-3 py-1.5 text-xs font-mono select-none flex items-center gap-2">
              <span className="text-indigo-600 dark:text-indigo-300 font-semibold">
                → {tc.function.name}
              </span>
              <span className="text-slate-400 truncate">
                {args && typeof args === "object"
                  ? Object.keys(args).map(k => `${k}=…`).join(" ")
                  : (tc.function.arguments || "").slice(0, 80)}
              </span>
            </summary>
            <div className="px-3 pb-3 space-y-2">
              {args && typeof args === "object" && !Array.isArray(args)
                ? Object.entries(args).map(([k, v]) => (
                    <ArgBlock key={k} name={k} value={v} />
                  ))
                : <pre className="text-xs font-mono whitespace-pre-wrap bg-slate-50 dark:bg-slate-800 p-2 rounded">{tc.function.arguments}</pre>}
            </div>
          </details>
        );
      })}
    </div>
  );
}

function ArgBlock({ name, value }: { name: string; value: unknown }) {
  const isStr = typeof value === "string";
  const display = isStr ? (value as string) : JSON.stringify(value, null, 2);
  const isMultiline = display.includes("\n") || display.length > 60;
  return (
    <div>
      <div className="text-[10px] uppercase text-slate-500 mb-0.5 flex items-center gap-2">
        <span>{name}</span>
        <button
          type="button"
          className="text-slate-400 hover:text-slate-700"
          onClick={() => navigator.clipboard.writeText(display)}
          title="copy"
        >
          📋
        </button>
      </div>
      {isMultiline
        ? <pre className="text-xs font-mono whitespace-pre-wrap bg-slate-50 dark:bg-slate-800 p-2 rounded max-h-64 overflow-auto">{display}</pre>
        : <code className="text-xs font-mono">{display}</code>}
    </div>
  );
}

function ToolResult({ msg, dur }: { msg: TraceMessage; dur?: number }) {
  const c = msg.content || "";
  const isLong = c.length > 800;
  return (
    <details open={!isLong}>
      <summary className="cursor-pointer text-xs select-none flex items-center gap-2">
        <span className="text-amber-700 dark:text-amber-300 font-semibold">← {msg.name}</span>
        <span className="text-slate-500">{c.length} chars</span>
        {dur !== undefined && (
          <span className="text-slate-400 font-mono">· {fmtMs(dur)}</span>
        )}
      </summary>
      <pre className="whitespace-pre-wrap text-xs font-mono mt-1 max-h-96 overflow-auto bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded p-2">{c || "(empty)"}</pre>
    </details>
  );
}

function countAsstUpTo(trace: TraceMessage[], i: number): number {
  let n = 0;
  for (let j = 0; j <= i; j++) {
    if (trace[j].role === "assistant" && trace[j].tool_calls?.length) n++;
  }
  return n;
}

export default function Trace() {
  const [params, setParams] = useSearchParams();
  const runName = params.get("run") ?? "";
  const pid     = params.get("pid") ?? "";

  const [runs,    setRuns]   = useState<RunSummary[]>([]);
  const [pids,    setPids]   = useState<AgentOutputEntry[]>([]);
  const [trace,   setTrace]  = useState<TraceMessage[] | null>(null);
  const [tcLog,   setTcLog]  = useState<ToolCallLog[]>([]);
  const [err,     setErr]    = useState<string | null>(null);
  const [loading, setLoading]= useState(false);

  useEffect(() => { api.runs().then(setRuns).catch(e => setErr(String(e))); }, []);

  // Reload problem list when run changes
  useEffect(() => {
    if (!runName) { setPids([]); return; }
    api.agentOutput(runName).then(setPids).catch(e => setErr(String(e)));
  }, [runName]);

  // Reload trace + tool_calls when (run, pid) change
  useEffect(() => {
    if (!runName || !pid) { setTrace(null); setTcLog([]); return; }
    setLoading(true); setErr(null);
    Promise.allSettled([
      api.trace(runName, pid),
      api.toolCalls(runName, pid),
    ]).then(([trc, tc]) => {
      if (trc.status === "fulfilled") setTrace(trc.value);
      else { setErr(String(trc.reason)); setTrace(null); }
      setTcLog(tc.status === "fulfilled" ? tc.value : []);
      setLoading(false);
    });
  }, [runName, pid]);

  // Per-message cumulative elapsed (sum of tool_calls.elapsed_ms up to
  // that point). Maps message index → cumulative ms.
  const cumByMsg = useMemo(() => {
    if (!trace || tcLog.length === 0) return new Map<number, number>();
    // Map step → total ms for that step
    const perStep = new Map<number, number>();
    for (const e of tcLog) {
      perStep.set(e.step, (perStep.get(e.step) ?? 0) + e.elapsed_ms);
    }
    // Walk trace messages, accumulate by assistant step counter
    const out = new Map<number, number>();
    let step = 0;
    let cum = 0;
    for (let i = 0; i < trace.length; i++) {
      const m = trace[i];
      if (m.role === "assistant" && m.tool_calls?.length) {
        step += 1;
        cum += perStep.get(step) ?? 0;
        out.set(i, cum);
      } else if (m.role === "tool") {
        out.set(i, cum);
      }
    }
    return out;
  }, [trace, tcLog]);

  // Tool durations indexed by tool_call_id (best effort: match by step + tool name).
  const durBySig = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of tcLog) {
      // Index by "step:tool" — collisions rare unless multi-tool-call per step.
      m.set(`${e.step}:${e.tool}`, e.elapsed_ms);
    }
    return m;
  }, [tcLog]);

  const updateParam = (k: string, v: string) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v); else next.delete(k);
    setParams(next, { replace: true });
  };

  const stats = useMemo(() => {
    if (!trace) return null;
    const roles: Record<string, number> = {};
    for (const m of trace) roles[m.role] = (roles[m.role] || 0) + 1;
    const ntcs = trace.reduce((n, m) => n + (m.tool_calls?.length ?? 0), 0);
    return { total: trace.length, roles, tool_calls: ntcs };
  }, [trace]);

  return (
    <div className="max-w-screen-xl mx-auto p-6 space-y-4">
      <div>
        <h2 className="text-2xl font-semibold mb-2">Trace</h2>
        <p className="text-sm text-slate-500">
          Agent conversation log for a single (run, problem).
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3 border-b border-slate-200 dark:border-slate-700 pb-3">
        <label className="flex flex-col text-xs uppercase text-slate-500">
          run
          <select
            className="mt-1 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm font-mono bg-white dark:bg-slate-800"
            value={runName}
            onChange={(e) => updateParam("run", e.target.value)}
          >
            <option value="">— select —</option>
            {runs.map((r) => (
              <option key={r.name} value={r.name}>{r.name}</option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-xs uppercase text-slate-500">
          problem
          <select
            className="mt-1 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm font-mono bg-white dark:bg-slate-800"
            value={pid}
            onChange={(e) => updateParam("pid", e.target.value)}
            disabled={!pids.length}
          >
            <option value="">— select —</option>
            {pids.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} {p.submitted ? "✓" : "·"} ({p.steps} steps)
              </option>
            ))}
          </select>
        </label>
        {stats && (
          <div className="text-xs text-slate-500 font-mono ml-auto">
            {stats.total} msgs · {stats.tool_calls} tool calls · {stats.roles.assistant ?? 0} asst · {stats.roles.tool ?? 0} tool
          </div>
        )}
      </div>

      {err && <pre className="text-red-600 text-sm">{err}</pre>}
      {loading && <p className="text-slate-500 text-sm">Loading…</p>}
      {!trace && !loading && !err && (
        <p className="text-slate-500 text-sm">Pick a run + problem above.</p>
      )}

      {trace && (
        <ol className="space-y-2">
          {trace.map((m, i) => {
            const cum = cumByMsg.get(i);
            return (
              <li
                key={i}
                className={`border-l-4 rounded px-4 py-2 ${ROLE_COLOR[m.role] ?? "border-slate-300 bg-slate-50"}`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] font-bold tracking-wider text-slate-500">
                    #{i} {ROLE_LABEL[m.role] ?? m.role.toUpperCase()}
                  </span>
                  {m.name && <span className="text-[10px] font-mono text-slate-500">{m.name}</span>}
                  {cum !== undefined && (
                    <span className="ml-auto text-[10px] font-mono text-slate-400">
                      cum {fmtMs(cum)}
                    </span>
                  )}
                </div>
                {m.role === "tool"
                  ? <ToolResult msg={m} dur={durBySig.get(`${countAsstUpTo(trace, i)}:${m.name ?? ""}`)} />
                  : <MessageContent content={m.content || ""} />}
                <ToolCalls msg={m} />
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
