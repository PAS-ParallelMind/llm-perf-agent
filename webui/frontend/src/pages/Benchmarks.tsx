import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, type Benchmarks as BenchT, type ProblemSpec } from "../api/client";

function TypeBadge({ prob }: { prob: ProblemSpec }) {
  const byteDet = prob.byte_deterministic !== false;
  const hasChecker = !!prob.checker;
  if (byteDet && !hasChecker) {
    return (
      <span className="px-2 py-0.5 text-[10px] font-mono rounded border border-emerald-500 text-emerald-700 dark:text-emerald-300">
        byte-deterministic
      </span>
    );
  }
  return (
    <span className="px-2 py-0.5 text-[10px] font-mono rounded border border-amber-500 text-amber-700 dark:text-amber-300">
      checker-required
    </span>
  );
}

function Section({
  title,
  language,
  args,
  code,
  defaultOpen = true,
}: {
  title: string;
  language?: string;
  args?: string;
  code: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border border-slate-200 dark:border-slate-700 rounded mb-4 overflow-hidden">
      <header
        className="flex items-center gap-3 px-3 py-2 bg-slate-100 dark:bg-slate-800 cursor-pointer select-none border-b border-slate-200 dark:border-slate-700"
        onClick={() => setOpen(!open)}
      >
        <span className="text-slate-400 text-xs">{open ? "▾" : "▸"}</span>
        <h3 className="text-sm font-semibold flex-1">{title}</h3>
        {language && (
          <span className="text-[10px] font-mono text-slate-500 px-1.5 py-0.5 bg-white dark:bg-slate-900 rounded border border-slate-300 dark:border-slate-600">
            {language}
          </span>
        )}
        {args && (
          <span className="text-[10px] font-mono text-slate-500">args: {args}</span>
        )}
      </header>
      {open && (
        code
          ? <CodeBlock code={code} />
          : <div className="px-3 py-2 text-xs italic text-slate-400">(empty)</div>
      )}
    </section>
  );
}

function CodeBlock({ code }: { code: string }) {
  return (
    <pre className="m-0 px-4 py-3 overflow-x-auto text-xs font-mono leading-relaxed bg-slate-900 text-slate-100">
      {code.split("\n").map((line, i) => (
        <div key={i}>
          <span className="inline-block w-8 text-right pr-3 text-slate-500 select-none">{i + 1}</span>
          <span>{line}</span>
        </div>
      ))}
    </pre>
  );
}

export default function Benchmarks() {
  const [bench, setBench] = useState<BenchT | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [params, setParams] = useSearchParams();
  const selPid = params.get("pid") ?? "";

  useEffect(() => {
    api.benchmarks().then(setBench).catch((e) => setErr(String(e)));
  }, []);

  const pids = useMemo(() => bench ? Object.keys(bench.problems).sort() : [], [bench]);
  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q || !bench) return pids;
    return pids.filter((pid) => {
      const p = bench.problems[pid];
      return (pid + " " + p.name).toLowerCase().includes(q);
    });
  }, [filter, pids, bench]);

  // Auto-select first problem if nothing chosen
  useEffect(() => {
    if (!selPid && filtered.length) {
      const next = new URLSearchParams(params);
      next.set("pid", filtered[0]);
      setParams(next, { replace: true });
    }
  }, [selPid, filtered, params, setParams]);

  if (err) return <div className="p-6 text-red-600 text-sm font-mono">{err}</div>;
  if (!bench) return <div className="p-6 text-slate-500">Loading…</div>;

  const prob = selPid ? bench.problems[selPid] : null;
  const pickPid = (pid: string) => {
    const next = new URLSearchParams(params);
    next.set("pid", pid);
    setParams(next, { replace: true });
  };

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <aside className="w-72 border-r border-slate-200 dark:border-slate-700 flex flex-col">
        <div className="px-3 py-2 border-b border-slate-200 dark:border-slate-700">
          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={`filter ${pids.length} problems…`}
            className="w-full text-sm px-2 py-1 border border-slate-300 dark:border-slate-600 rounded bg-white dark:bg-slate-800"
          />
        </div>
        <ul className="flex-1 overflow-y-auto">
          {filtered.map((pid) => {
            const p = bench.problems[pid];
            const isActive = pid === selPid;
            const byteDet = p.byte_deterministic !== false;
            return (
              <li
                key={pid}
                onClick={() => pickPid(pid)}
                className={`px-3 py-1.5 cursor-pointer border-b border-slate-100 dark:border-slate-800 flex items-baseline gap-2 ${
                  isActive
                    ? "bg-indigo-50 dark:bg-indigo-900/30 border-l-4 border-l-indigo-500 pl-2"
                    : "hover:bg-slate-50 dark:hover:bg-slate-800/40"
                }`}
              >
                <span className="font-mono text-xs text-indigo-700 dark:text-indigo-300">{pid}</span>
                <span className="flex-1 text-sm truncate">{p.name}</span>
                <span
                  className={`text-[9px] font-mono px-1 py-0.5 rounded border ${
                    byteDet
                      ? "border-emerald-400 text-emerald-700 dark:text-emerald-400"
                      : "border-amber-400 text-amber-700 dark:text-amber-400"
                  }`}
                >
                  {byteDet ? "byte" : "chk"}
                </span>
              </li>
            );
          })}
        </ul>
      </aside>

      {/* Detail pane */}
      <main className="flex-1 overflow-y-auto p-6">
        {!prob ? (
          <p className="text-slate-500 text-sm">Pick a problem from the sidebar.</p>
        ) : (
          <div className="max-w-5xl">
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-2xl font-mono text-indigo-700 dark:text-indigo-300">{selPid}</h2>
              <TypeBadge prob={prob} />
            </div>
            <div className="text-base mb-4 text-slate-700 dark:text-slate-200">{prob.name}</div>

            <pre className="whitespace-pre-wrap text-sm leading-relaxed bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded p-4 mb-6">
              {prob.description}
            </pre>

            <Section
              title="Reference (serial)"
              language={prob.reference.language}
              code={prob.reference.code}
            />
            <Section
              title="gen_input"
              language={prob.gen_input.language}
              args={(prob.gen_input.default_args ?? []).join(" ")}
              code={prob.gen_input.code}
            />
            <Section
              title="Checker"
              language={prob.checker?.language}
              code={prob.checker?.code ?? ""}
              defaultOpen={false}
            />
          </div>
        )}
      </main>
    </div>
  );
}
