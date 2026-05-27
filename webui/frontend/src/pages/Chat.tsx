import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type RunSummary,
  type TraceMessage,
  type TurnEvent,
} from "../api/client";
import Markdown from "../components/Markdown";
import NewSessionModal from "../components/NewSessionModal";

// Conversational view of a chat session: markdown-rendered user /
// assistant turns with tool calls + results folded behind disclosures.
// While a turn is in flight, a "live" bubble below the user message
// renders step / tool events as they stream from /chat_stream — so the
// user sees exactly which tool the agent is calling, not just a vague
// "thinking…" indicator.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tryParseJSON(s: string): unknown {
  try { return JSON.parse(s); } catch { return null; }
}

function stripPreviousAnalysis(content: string): string {
  // The loop pre-pends "[previous analysis] …" reasoning to the user's
  // message when --reasoning is on. Hide it from the chat view; the raw
  // text is still visible in the Trace page.
  return content.replace(/^\[previous analysis\][\s\S]*?(?:\n\n|\n*$)/, "");
}

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms - m * 60_000) / 1000);
  return `${m}m${s.toString().padStart(2, "0")}s`;
}

function summarizeArgs(raw: string): string {
  const parsed = tryParseJSON(raw);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return Object.entries(parsed as Record<string, unknown>)
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
      .join(" ")
      .slice(0, 140);
  }
  return (raw || "").slice(0, 140);
}

// ---------------------------------------------------------------------------
// Live timeline state
// ---------------------------------------------------------------------------

type LiveTool = {
  id: string;
  name: string;
  arguments: string;
  status: "running" | "done";
  elapsedMs?: number;
  resultChars?: number;
  preview?: string;
};

type LiveStep = {
  step: number;
  assistantText?: string;
  llmElapsedMs?: number;
  tools: LiveTool[];
};

type LiveState = {
  steps: LiveStep[];
  synthesizing: boolean;
};

const EMPTY_LIVE: LiveState = { steps: [], synthesizing: false };

function applyEvent(state: LiveState, ev: TurnEvent): LiveState {
  switch (ev.type) {
    case "ready":
      return EMPTY_LIVE;
    case "step_start":
      return {
        ...state,
        steps: [...state.steps, { step: ev.step, tools: [] }],
      };
    case "assistant": {
      const steps = state.steps.slice();
      const idx = steps.findIndex(s => s.step === ev.step);
      if (idx >= 0) {
        steps[idx] = {
          ...steps[idx],
          assistantText: ev.content,
          llmElapsedMs: ev.llm_elapsed_ms,
        };
      }
      return { ...state, steps };
    }
    case "tool_start": {
      const steps = state.steps.slice();
      const idx = steps.findIndex(s => s.step === ev.step);
      if (idx >= 0) {
        steps[idx] = {
          ...steps[idx],
          tools: [...steps[idx].tools, {
            id:        ev.id,
            name:      ev.name,
            arguments: ev.arguments,
            status:    "running",
          }],
        };
      }
      return { ...state, steps };
    }
    case "tool_done": {
      const steps = state.steps.slice();
      const idx = steps.findIndex(s => s.step === ev.step);
      if (idx >= 0) {
        steps[idx] = {
          ...steps[idx],
          tools: steps[idx].tools.map(t =>
            t.id === ev.id
              ? { ...t, status: "done",
                  elapsedMs:   ev.elapsed_ms,
                  resultChars: ev.result_chars,
                  preview:     ev.preview }
              : t),
        };
      }
      return { ...state, steps };
    }
    case "synthesis_start":
      return { ...state, synthesizing: true };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Presentational components
// ---------------------------------------------------------------------------

function ToolCallBlock({ tc, result }: {
  tc: NonNullable<TraceMessage["tool_calls"]>[number];
  result?: TraceMessage;
}) {
  const args = tryParseJSON(tc.function.arguments);
  const argSummary = summarizeArgs(tc.function.arguments);
  const out = result?.content ?? "";
  return (
    <details className="border border-slate-200 dark:border-slate-700 rounded bg-slate-50 dark:bg-slate-800/40 my-1">
      <summary className="cursor-pointer px-3 py-1.5 text-xs font-mono select-none flex items-center gap-2">
        <span className="text-indigo-600 dark:text-indigo-300 font-semibold">
          ⚒ {tc.function.name}
        </span>
        <span className="text-slate-500 truncate">{argSummary}</span>
        {out && (
          <span className="ml-auto text-slate-400">{out.length} chars</span>
        )}
      </summary>
      <div className="px-3 pb-3 space-y-2 text-xs font-mono">
        <div>
          <div className="text-[10px] uppercase text-slate-500 mb-0.5">arguments</div>
          <pre className="whitespace-pre-wrap bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded p-2 max-h-48 overflow-auto">
            {args !== null ? JSON.stringify(args, null, 2) : tc.function.arguments}
          </pre>
        </div>
        <div>
          <div className="text-[10px] uppercase text-slate-500 mb-0.5">
            result {result ? "" : "(pending)"}
          </div>
          <pre className="whitespace-pre-wrap bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded p-2 max-h-96 overflow-auto">
            {out || "(no output)"}
          </pre>
        </div>
      </div>
    </details>
  );
}

function LiveToolRow({ tool }: { tool: LiveTool }) {
  const argSummary = summarizeArgs(tool.arguments);
  return (
    <details className="border border-slate-200 dark:border-slate-700 rounded bg-slate-50 dark:bg-slate-800/40 my-1">
      <summary className="cursor-pointer px-3 py-1.5 text-xs font-mono select-none flex items-center gap-2">
        <span
          className={
            "inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] " +
            (tool.status === "running"
              ? "bg-amber-400 text-amber-900 animate-pulse"
              : "bg-emerald-500 text-white")
          }
          title={tool.status}
        >
          {tool.status === "running" ? "…" : "✓"}
        </span>
        <span className="text-indigo-600 dark:text-indigo-300 font-semibold">
          ⚒ {tool.name}
        </span>
        <span className="text-slate-500 truncate">{argSummary}</span>
        {tool.status === "done" && (
          <span className="ml-auto text-slate-400 flex items-center gap-2">
            {tool.elapsedMs !== undefined && <span>{fmtMs(tool.elapsedMs)}</span>}
            {tool.resultChars !== undefined && <span>{tool.resultChars} chars</span>}
          </span>
        )}
      </summary>
      {tool.preview && (
        <pre className="text-xs font-mono whitespace-pre-wrap bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded m-2 p-2 max-h-64 overflow-auto">
          {tool.preview}
        </pre>
      )}
    </details>
  );
}

function LiveAssistantBubble({ state }: { state: LiveState }) {
  const empty = state.steps.length === 0;
  return (
    <div className="flex justify-start">
      <div className="max-w-3xl bg-white dark:bg-slate-800/60 border border-indigo-200 dark:border-indigo-800 rounded-lg px-4 py-2 w-full">
        <div className="text-[10px] uppercase text-indigo-600 dark:text-indigo-300 font-bold mb-1 flex items-center gap-2">
          <span>assistant</span>
          <span className="inline-block w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
          <span className="text-slate-500 normal-case font-normal">
            {empty ? "connecting…" : `step ${state.steps[state.steps.length - 1].step} · live`}
          </span>
          {state.synthesizing && (
            <span className="text-amber-600 dark:text-amber-300 normal-case font-normal">
              · step budget hit — synthesizing final answer
            </span>
          )}
        </div>
        {empty && (
          <div className="text-sm text-slate-500 italic">
            Waiting for the model's first step…
          </div>
        )}
        {state.steps.map((s) => {
          const text = (s.assistantText ?? "").trim();
          return (
            <div key={s.step} className="mt-1">
              {text && <Markdown>{stripPreviousAnalysis(text)}</Markdown>}
              {s.tools.map(t => <LiveToolRow key={t.id} tool={t} />)}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function UserBubble({ content }: { content: string }) {
  const cleaned = stripPreviousAnalysis(content);
  return (
    <div className="flex justify-end">
      <div className="max-w-3xl bg-emerald-100 dark:bg-emerald-900/30 border border-emerald-300 dark:border-emerald-700 rounded-lg px-4 py-2">
        <div className="text-[10px] uppercase text-emerald-700 dark:text-emerald-300 font-bold mb-1">you</div>
        <Markdown>{cleaned || "(empty)"}</Markdown>
      </div>
    </div>
  );
}

function AssistantBubble({
  content, toolCalls, resultsById,
}: {
  content: string;
  toolCalls?: TraceMessage["tool_calls"];
  resultsById: Map<string, TraceMessage>;
}) {
  const hasText = content && content.trim().length > 0;
  return (
    <div className="flex justify-start">
      <div className="max-w-3xl bg-white dark:bg-slate-800/60 border border-indigo-200 dark:border-indigo-800 rounded-lg px-4 py-2 w-full">
        <div className="text-[10px] uppercase text-indigo-600 dark:text-indigo-300 font-bold mb-1">assistant</div>
        {hasText && <Markdown>{content}</Markdown>}
        {toolCalls?.map(tc => (
          <ToolCallBlock key={tc.id} tc={tc} result={resultsById.get(tc.id)} />
        ))}
      </div>
    </div>
  );
}

function SystemBubble({ content }: { content: string }) {
  return (
    <details className="text-xs text-slate-500">
      <summary className="cursor-pointer select-none">
        system prompt ({content.length} chars)
      </summary>
      <pre className="whitespace-pre-wrap mt-1 font-mono text-[11px] border-l-2 border-slate-300 dark:border-slate-600 pl-3">
        {content}
      </pre>
    </details>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Chat() {
  const [runs, setRuns]       = useState<RunSummary[]>([]);
  const [runName, setRunName] = useState<string>("");
  const [messages, setMsgs]   = useState<TraceMessage[]>([]);
  const [draft, setDraft]     = useState<string>("");
  const [sending, setSending] = useState(false);
  const [showSys, setShowSys] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [err, setErr]         = useState<string | null>(null);
  const [lastInfo, setInfo]   = useState<string>("");
  const [live, setLive]       = useState<LiveState>(EMPTY_LIVE);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Initial session list. Pick the most recent so the page is never empty.
  useEffect(() => {
    refreshRuns().then(rs => {
      if (rs.length && !runName) setRunName(rs[0].name);
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Load the selected session's trace whenever it changes. Cancel any
  // in-flight stream from a previous session so its events don't leak in.
  useEffect(() => {
    abortRef.current?.abort();
    setLive(EMPTY_LIVE);
    setSending(false);
    if (!runName) { setMsgs([]); return; }
    setErr(null);
    api.trace(runName)
      .then(setMsgs)
      .catch(e => { setErr(String(e)); setMsgs([]); });
  }, [runName]);

  // Auto-scroll to the bottom whenever new content lands. The dep on
  // live.steps.length + live tools' status changes covers in-flight tool
  // events, not just message-list growth.
  const liveSig = useMemo(
    () => live.steps.map(s => `${s.step}:${s.tools.length}:${s.tools.map(t => t.status).join("")}`).join("|"),
    [live],
  );
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, sending, liveSig]);

  async function refreshRuns(): Promise<RunSummary[]> {
    try {
      const rs = await api.runs();
      setRuns(rs);
      return rs;
    } catch (e) {
      setErr(String(e));
      return [];
    }
  }

  async function send() {
    const msg = draft.trim();
    if (!msg || !runName || sending) return;
    setSending(true); setErr(null); setInfo(""); setLive(EMPTY_LIVE);
    setMsgs(prev => [...prev, { role: "user", content: msg }]);
    setDraft("");

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      await api.chatStream(runName, msg, (ev) => {
        if (ev.type === "final") {
          setMsgs(ev.messages);
          setLive(EMPTY_LIVE);
          setInfo(`turn ${ev.turns} · ${ev.steps} step(s) · ${ev.elapsed_s}s`
                  + (ev.truncated ? " · truncated" : ""));
          refreshRuns();
        } else if (ev.type === "error") {
          setErr(ev.error);
        } else {
          setLive(prev => applyEvent(prev, ev));
        }
      }, ctrl.signal);
    } catch (e) {
      // AbortError is expected when the user switches sessions mid-turn.
      const name = (e as Error)?.name;
      if (name !== "AbortError") setErr(String(e));
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      setSending(false);
    }
  }

  async function resetSession() {
    if (!runName) return;
    if (!confirm(`Reset conversation in ${runName}? Tool outputs on disk are kept.`)) return;
    try {
      await api.reset(runName);
      const trace = await api.trace(runName);
      setMsgs(trace);
      setInfo("conversation reset");
    } catch (e) {
      setErr(String(e));
    }
  }

  // Pair each assistant message with its tool results by tool_call_id.
  // Standalone tool messages are surfaced inside the assistant bubble that
  // requested them — don't render them again at the top level.
  const view = useMemo(() => {
    const resultsById = new Map<string, TraceMessage>();
    for (const m of messages) {
      if (m.role === "tool" && m.tool_call_id) resultsById.set(m.tool_call_id, m);
    }
    return { items: messages, resultsById };
  }, [messages]);

  const currentRun = runs.find(r => r.name === runName);

  return (
    <div className="max-w-screen-xl mx-auto p-6 flex flex-col h-full gap-4">
      {/* Header: session picker + actions */}
      <div className="flex flex-wrap items-end gap-3 border-b border-slate-200 dark:border-slate-700 pb-3">
        <label className="flex flex-col text-xs uppercase text-slate-500">
          session
          <select
            className="mt-1 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm font-mono bg-white dark:bg-slate-800 min-w-[18rem]"
            value={runName}
            onChange={e => setRunName(e.target.value)}
          >
            <option value="">— select —</option>
            {runs.map(r => (
              <option key={r.name} value={r.name}>
                {r.name}
                {r.turns !== undefined ? ` · ${r.turns}t / ${r.total_steps ?? 0}s` : ""}
              </option>
            ))}
          </select>
        </label>
        {currentRun?.agent_model && (
          <div className="text-xs text-slate-500 font-mono">
            <span className="uppercase block text-[10px]">model</span>
            {currentRun.agent_model}
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button
            type="button"
            className="text-xs px-2 py-1 border border-slate-300 dark:border-slate-600 rounded hover:bg-slate-100 dark:hover:bg-slate-800"
            onClick={() => setShowSys(s => !s)}
          >{showSys ? "hide" : "show"} system</button>
          <button
            type="button"
            className="text-xs px-2 py-1 border border-slate-300 dark:border-slate-600 rounded hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-50"
            onClick={resetSession}
            disabled={!runName || sending}
          >reset</button>
          <button
            type="button"
            className="text-xs px-2 py-1 border border-indigo-400 text-indigo-700 dark:text-indigo-200 rounded hover:bg-indigo-50 dark:hover:bg-indigo-900/40"
            onClick={() => setShowNew(true)}
          >+ new session</button>
        </div>
      </div>

      {err && <pre className="text-red-600 text-xs whitespace-pre-wrap">{err}</pre>}

      {/* Conversation */}
      <div className="flex-1 overflow-auto space-y-3 pr-1">
        {!runName && (
          <p className="text-slate-500 text-sm">
            Pick a session above, or click <em>+ new session</em>.
          </p>
        )}

        {runName && view.items.length === 0 && !err && !sending && (
          <p className="text-slate-500 text-sm">Empty session — say hello below.</p>
        )}

        {view.items.map((m, i) => {
          if (m.role === "system") {
            return showSys
              ? <SystemBubble key={i} content={m.content ?? ""} />
              : null;
          }
          if (m.role === "user") {
            return <UserBubble key={i} content={m.content ?? ""} />;
          }
          if (m.role === "assistant") {
            return (
              <AssistantBubble
                key={i}
                content={stripPreviousAnalysis(m.content ?? "")}
                toolCalls={m.tool_calls}
                resultsById={view.resultsById}
              />
            );
          }
          return null;
        })}

        {sending && <LiveAssistantBubble state={live} />}

        <div ref={bottomRef} />
      </div>

      {/* Composer */}
      {runName && (
        <div className="border-t border-slate-200 dark:border-slate-700 pt-3">
          {lastInfo && (
            <div className="text-[11px] text-slate-500 mb-1 font-mono">{lastInfo}</div>
          )}
          <div className="flex gap-2 items-end">
            <textarea
              className="flex-1 border border-slate-300 dark:border-slate-600 rounded px-3 py-2 text-sm bg-white dark:bg-slate-800 font-mono resize-y min-h-[3rem] max-h-64"
              placeholder="Message the agent… (Enter to send, Shift+Enter for newline)"
              value={draft}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              disabled={sending}
              rows={2}
            />
            <button
              type="button"
              className="px-4 py-2 text-sm rounded bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50"
              onClick={send}
              disabled={sending || !draft.trim()}
            >Send</button>
          </div>
        </div>
      )}

      {showNew && (
        <NewSessionModal
          onClose={() => setShowNew(false)}
          onCreated={async (run) => {
            setShowNew(false);
            await refreshRuns();
            setRunName(run.name);
          }}
        />
      )}
    </div>
  );
}
