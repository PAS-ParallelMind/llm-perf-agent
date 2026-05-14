import type { RunSummary } from "../api/client";

export default function RunPicker({
  runs,
  selected,
  onChange,
}: {
  runs: RunSummary[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
}) {
  const toggle = (name: string) => {
    const next = new Set(selected);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    onChange(next);
  };

  return (
    <div className="flex flex-wrap gap-2">
      {runs.map((r) => {
        const isOn = selected.has(r.name);
        const pass = r.has_eval && r.n_pass !== undefined
          ? `${r.n_pass}/${r.n_total}`
          : `${r.n_submitted ?? "?"}/${r.n_total ?? "?"} sub`;
        return (
          <button
            key={r.name}
            type="button"
            onClick={() => toggle(r.name)}
            className={`px-3 py-1.5 text-xs rounded border transition-colors ${
              isOn
                ? "bg-indigo-600 text-white border-indigo-600"
                : "bg-white text-slate-700 border-slate-300 hover:border-indigo-400 dark:bg-slate-800 dark:text-slate-200"
            }`}
            title={r.model_name ?? ""}
          >
            <span className="font-mono">{r.name}</span>
            <span className="ml-2 opacity-80">{pass}</span>
          </button>
        );
      })}
    </div>
  );
}
