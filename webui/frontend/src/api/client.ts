// Tiny fetch wrappers around the FastAPI backend at /api/*.
// Vite dev server proxies /api → :8080; in production the same backend
// serves the built frontend so /api is same-origin.

export type RunSummary = {
  name: string;
  modified_at: string;
  started_at?: string;
  agent_model?: string;
  turns?: number;
  total_steps?: number;
  total_elapsed_s?: number;
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

export type ToolCallLog = {
  turn?: number;
  step: number;
  tool: string;
  arguments: string;
  result: string;
  elapsed_ms: number;
};

export type ToolSchema = {
  type: string;
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
  };
};

// One snapshot of the exact payload sent to the model on a single LLM call.
export type LlmRequest = {
  turn: number;
  step: number;
  elapsed_ms?: number;
  model: string;
  messages: TraceMessage[];
  tools: ToolSchema[];
  params: Record<string, unknown>;
};

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${resp.status} ${path}: ${await resp.text()}`);
  return resp.json();
}

export const api = {
  runs:        () => get<RunSummary[]>("/api/runs"),
  run:         (name: string) => get<RunSummary>(`/api/runs/${encodeURIComponent(name)}`),
  trace:       (name: string) => get<TraceMessage[]>(`/api/runs/${encodeURIComponent(name)}/trace`),
  toolCalls:   (name: string) => get<ToolCallLog[]>(`/api/runs/${encodeURIComponent(name)}/tool_calls`),
  llmRequests: (name: string) => get<LlmRequest[]>(`/api/runs/${encodeURIComponent(name)}/llm_requests`),
};
