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

// Live events emitted by the agent loop and streamed over SSE.
// `final` carries the same fields as ChatResponse; `error` carries `error`.
export type TurnEvent =
  | { type: "ready" }
  | { type: "step_start"; step: number; max_steps: number }
  | {
      type: "assistant";
      step: number;
      content: string;
      tool_calls: NonNullable<TraceMessage["tool_calls"]>;
      has_tool_calls: boolean;
      llm_elapsed_ms: number;
    }
  | {
      type: "tool_start";
      step: number;
      id: string;
      name: string;
      arguments: string;
    }
  | {
      type: "tool_done";
      step: number;
      id: string;
      name: string;
      elapsed_ms: number;
      preview: string;
      result_chars: number;
    }
  | { type: "synthesis_start"; reason: string; max_steps: number }
  | ({ type: "final" } & ChatResponse)
  | { type: "error"; error: string };

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

// POST + SSE: post the message and stream events as they arrive. Browser
// EventSource doesn't support POST, so we drive fetch's response stream
// ourselves and parse the "data: ...\n\n" framing manually.
//
// Returns a promise that resolves when the stream ends. `onEvent` is
// called with each parsed event in order. Throws on HTTP errors.
async function chatStream(
  name: string,
  message: string,
  onEvent: (ev: TurnEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(
    `/api/runs/${encodeURIComponent(name)}/chat_stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal,
    },
  );
  if (!resp.ok || !resp.body) {
    throw new Error(`${resp.status}: ${await resp.text()}`);
  }

  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. Handle either \n\n or
    // \r\n\r\n to be tolerant of proxies that normalize line endings.
    let sep: number;
    while (
      (sep = (() => {
        const a = buf.indexOf("\n\n");
        const b = buf.indexOf("\r\n\r\n");
        if (a === -1) return b;
        if (b === -1) return a;
        return Math.min(a, b);
      })()) !== -1
    ) {
      const skip = buf.startsWith("\r\n\r\n", sep) ? 4 : 2;
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + skip);

      // Each frame may have multiple `data:` lines; concatenate them.
      const data = frame
        .split(/\r?\n/)
        .filter(l => l.startsWith("data:"))
        .map(l => l.slice(5).replace(/^ /, ""))
        .join("\n");
      if (!data) continue;
      try {
        onEvent(JSON.parse(data) as TurnEvent);
      } catch {
        // Malformed frame — surface as an error event so the UI can react.
        onEvent({ type: "error", error: `bad SSE frame: ${data.slice(0, 200)}` });
      }
    }
  }
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
  chatStream,
  reset:       (name: string) =>
    post<RunSummary>(`/api/runs/${encodeURIComponent(name)}/reset`, {}),
};
