// Tiny fetch wrappers around the FastAPI backend at /api/*.
// Vite dev server proxies /api → :8080; in production the same backend
// serves the built frontend so /api is same-origin.

export type RunSummary = {
  name: string;
  modified_at: string;
  model_name?: string;
  workers?: number;
  n_total?: number;
  n_submitted?: number;
  agent_output_path?: string;
  has_eval: boolean;
  n_pass?: number;
};

export type ProblemSpec = {
  id: string;
  name: string;
  byte_deterministic?: boolean;
  description: string;
  reference: { language: string; code: string };
  gen_input: { language: string; code: string; default_args?: string[] };
  checker?: { language: string; code: string } | null;
};

export type Benchmarks = { problems: Record<string, ProblemSpec> };

export type AgentOutputEntry = {
  id: string;
  code: string;
  notes?: string;
  submitted: boolean;
  steps: number;
  elapsed_s: number;
  error?: string | null;
  metadata?: Record<string, unknown>;
};

export type ValidationCase = {
  input: string;
  status: string;
  ref_elapsed_s?: number;
  cand_elapsed_s?: number;
  ref_bytes?: number;
  cand_bytes?: number;
  detail?: string;
  checker_valid?: boolean | null;
  checker_reason?: string | null;
  llm_verdict?: string | null;
  llm_reasoning?: string | null;
};

export type EvalEntry = {
  id: string;
  submitted: boolean;
  validation?: {
    summary: {
      total: number;
      pass_byte: number;
      pass_checker: number;
      pass_llm: number;
      fail: number;
      fail_ref: number;
      fail_cand: number;
      error_llm: number;
      build_fail_ref: number;
      build_fail_cand: number;
      pass_rate: string;
    };
    cases: ValidationCase[];
  } | null;
  speedup?: {
    ref_wall_ms?: number | null;
    ref_compute_ms?: number | null;
    cand_wall_ms?: number | null;
    cand_kernel_ms?: number | null;
    speedup_e2e?: number | null;
    speedup_kernel?: number | null;
  } | null;
};

export type TraceMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content?: string;
  tool_calls?: Array<{
    id: string;
    type: string;
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
  name?: string;
};

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${resp.status} ${path}: ${await resp.text()}`);
  return resp.json();
}

async function getText(path: string): Promise<string> {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${resp.status} ${path}: ${await resp.text()}`);
  return resp.text();
}

export type ToolCallLog = {
  step: number;
  tool: string;
  arguments: string;
  result: string;
  elapsed_ms: number;
};

export const api = {
  benchmarks:    () => get<Benchmarks>("/api/benchmarks"),
  runs:          () => get<RunSummary[]>("/api/runs"),
  run:           (name: string) => get<RunSummary>(`/api/runs/${encodeURIComponent(name)}`),
  agentOutput:   (name: string) => get<AgentOutputEntry[]>(`/api/runs/${encodeURIComponent(name)}/agent_output`),
  evalResults:   (name: string) => get<EvalEntry[]>(`/api/runs/${encodeURIComponent(name)}/eval_results`),
  trace:         (name: string, pid: string) =>
                   get<TraceMessage[]>(`/api/runs/${encodeURIComponent(name)}/batch/${pid}/trace`),
  toolCalls:     (name: string, pid: string) =>
                   get<ToolCallLog[]>(`/api/runs/${encodeURIComponent(name)}/batch/${pid}/tool_calls`),
  code:          (name: string, pid: string) =>
                   getText(`/api/runs/${encodeURIComponent(name)}/batch/${pid}/code`),
};
