import { useEffect, useState } from "react";
import { api, type ModelPreset, type NewSessionRequest, type RunSummary } from "../api/client";

type Props = {
  onCreated: (run: RunSummary) => void;
  onClose: () => void;
};

// Modal-style overlay for creating a new chat session. Loads available
// presets from /api/presets, lets the user pick one, then exposes the
// underlying fields for last-mile overrides (base_url, reasoning, etc.).
export default function NewSessionModal({ onCreated, onClose }: Props) {
  const [presets, setPresets]   = useState<ModelPreset[]>([]);
  const [presetIdx, setIdx]     = useState(0);
  const [form, setForm]         = useState<NewSessionRequest | null>(null);
  const [busy, setBusy]         = useState(false);
  const [err, setErr]           = useState<string | null>(null);

  useEffect(() => {
    api.presets()
      .then(ps => {
        setPresets(ps);
        if (ps.length) setForm(presetToForm(ps[0]));
      })
      .catch(e => setErr(String(e)));
  }, []);

  function pickPreset(i: number) {
    setIdx(i);
    if (presets[i]) setForm(presetToForm(presets[i]));
  }

  function setModel<K extends keyof NewSessionRequest["model"]>(
    k: K,
    v: NewSessionRequest["model"][K],
  ) {
    setForm(f => (f ? { ...f, model: { ...f.model, [k]: v } } : f));
  }

  async function submit() {
    if (!form) return;
    setBusy(true); setErr(null);
    try {
      const run = await api.newSession(form);
      onCreated(run);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!form) {
    return (
      <Overlay onClose={onClose}>
        <div className="text-sm text-slate-500">
          {err ? <span className="text-red-600">{err}</span> : "Loading presets…"}
        </div>
      </Overlay>
    );
  }

  return (
    <Overlay onClose={onClose}>
      <h2 className="text-lg font-semibold mb-3">New chat session</h2>

      <Field label="preset">
        <select
          className="border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm bg-white dark:bg-slate-800 w-full"
          value={presetIdx}
          onChange={e => pickPreset(Number(e.target.value))}
        >
          {presets.map((p, i) => (
            <option key={p.file + i} value={i}>{p.label}</option>
          ))}
        </select>
      </Field>

      <Field label="session name (optional)">
        <input
          type="text"
          placeholder="chat-YYYYMMDD-HHMMSS"
          className="border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm bg-white dark:bg-slate-800 w-full font-mono"
          value={form.name ?? ""}
          onChange={e => setForm({ ...form, name: e.target.value })}
        />
      </Field>

      <Field label="model name">
        <input
          type="text"
          className="border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm bg-white dark:bg-slate-800 w-full font-mono"
          value={form.model.name}
          onChange={e => setModel("name", e.target.value)}
        />
      </Field>

      <Field label="base URL">
        <input
          type="text"
          className="border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm bg-white dark:bg-slate-800 w-full font-mono"
          value={form.model.base_url ?? ""}
          onChange={e => setModel("base_url", e.target.value)}
        />
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field label="temperature">
          <input
            type="number" step="0.1" min="0"
            className="border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm bg-white dark:bg-slate-800 w-full font-mono"
            value={form.model.temperature ?? 0}
            onChange={e => setModel("temperature", Number(e.target.value))}
          />
        </Field>
        <Field label="max steps">
          <input
            type="number" min="1"
            className="border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm bg-white dark:bg-slate-800 w-full font-mono"
            value={form.max_steps ?? 20}
            onChange={e => setForm({ ...form, max_steps: Number(e.target.value) })}
          />
        </Field>
        <Field label="max output tokens">
          <input
            type="number" min="1"
            className="border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm bg-white dark:bg-slate-800 w-full font-mono"
            value={form.model.max_output_tokens ?? 4096}
            onChange={e => setModel("max_output_tokens", Number(e.target.value))}
          />
        </Field>
        <Field label="context window">
          <input
            type="number" min="1024"
            className="border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm bg-white dark:bg-slate-800 w-full font-mono"
            value={form.model.max_model_len ?? 32768}
            onChange={e => setModel("max_model_len", Number(e.target.value))}
          />
        </Field>
      </div>

      <Field label="">
        <label className="inline-flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={!!form.model.reasoning}
            onChange={e => setModel("reasoning", e.target.checked)}
          />
          enable reasoning (model returns separate reasoning tokens)
        </label>
      </Field>

      {err && <pre className="text-red-600 text-xs whitespace-pre-wrap mb-2">{err}</pre>}

      <div className="flex gap-2 justify-end mt-2">
        <button
          type="button"
          className="px-3 py-1.5 text-sm rounded border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800"
          onClick={onClose}
          disabled={busy}
        >Cancel</button>
        <button
          type="button"
          className="px-3 py-1.5 text-sm rounded bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50"
          onClick={submit}
          disabled={busy || !form.model.name.trim()}
        >{busy ? "Creating…" : "Create session"}</button>
      </div>
    </Overlay>
  );
}

function presetToForm(p: ModelPreset): NewSessionRequest {
  return {
    name: "",
    max_steps: p.agent.max_steps ?? 20,
    system_prompt:      p.system_prompt ?? null,
    system_prompt_file: p.system_prompt_file ?? null,
    model: {
      name:              p.agent.model.name,
      base_url:          p.agent.model.base_url ?? "http://localhost:8000/v1",
      api_key:           p.agent.model.api_key ?? "EMPTY",
      temperature:       p.agent.model.temperature ?? 0,
      max_output_tokens: p.agent.model.max_output_tokens ?? 4096,
      max_model_len:     p.agent.model.max_model_len ?? 32768,
      reasoning:         !!p.agent.model.reasoning,
    },
  };
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block text-xs uppercase text-slate-500 mb-2">
      {label && <span className="block mb-1">{label}</span>}
      {children}
    </label>
  );
}

function Overlay({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 bg-slate-900/60 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-slate-900 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 p-5 w-full max-w-md max-h-[90vh] overflow-auto"
        onClick={e => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}
