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

// One model preset — a snapshot of a chat-config YAML in the harness root.
// The new-session form pre-fills from one of these.
export type ModelPreset = {
  file: string;
  label: string;
  agent: {
    model: {
      name: string;
      base_url?: string;
      api_key?: string;
      temperature?: number;
      max_output_tokens?: number;
      max_model_len?: number;
      reasoning?: boolean;
    };
    max_steps?: number;
  };
  system_prompt?: string | null;
  system_prompt_file?: string | null;
};

// Payload the UI POSTs to /api/sessions to create a new session.
export type NewSessionRequest = {
  name?: string;
  max_steps?: number;
  system_prompt?: string | null;
  system_prompt_file?: string | null;
  model: ModelPreset["agent"]["model"];
};

export type ChatResponse = {
  reply: string;
  steps: number;
  elapsed_s: number;
  truncated: boolean;
  messages: TraceMessage[];
  turns: number;
};

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${resp.status} ${path}: ${await resp.text()}`);
  return resp.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`${resp.status} ${path}: ${await resp.text()}`);
  return resp.json();
}

export const api = {
  runs:        () => get<RunSummary[]>("/api/runs"),
  run:         (name: string) => get<RunSummary>(`/api/runs/${encodeURIComponent(name)}`),
  trace:       (name: string) => get<TraceMessage[]>(`/api/runs/${encodeURIComponent(name)}/trace`),
  toolCalls:   (name: string) => get<ToolCallLog[]>(`/api/runs/${encodeURIComponent(name)}/tool_calls`),
  llmRequests: (name: string) => get<LlmRequest[]>(`/api/runs/${encodeURIComponent(name)}/llm_requests`),

  presets:     () => get<ModelPreset[]>("/api/presets"),
  newSession:  (req: NewSessionRequest) => post<RunSummary>("/api/sessions", req),
  chat:        (name: string, message: string) =>
    post<ChatResponse>(`/api/runs/${encodeURIComponent(name)}/chat`, { message }),
  reset:       (name: string) =>
    post<RunSummary>(`/api/runs/${encodeURIComponent(name)}/reset`, {}),
};
