# cc-style CUDA Agent

A lightweight Claude-Code-style agentic workflow in Python for writing and
debugging parallel code (CUDA / MPI / OpenMP). Backed by a **vLLM** server
(OpenAI-compatible API).

## Layout

```
agent/
  main.py        # interactive CLI entry
  engine.py      # vLLM (OpenAI) client wrapper
  loop.py        # agentic loop: model <-> tool calls
  memory.py      # markdown-file memory store (CLAUDE.md-style)
  prompts.py     # system prompt for parallel-code work
  tools/
    base.py      # @tool registry + JSON schema export
    fs.py        # read / write / edit / glob / grep
    bash.py      # sandboxed shell exec
    parallel.py  # nvcc / mpicc / OpenMP build + run helpers
memory/          # persistent memory dir (auto-created)
```

## Quick start

```bash
# 1. start a vLLM server (any tool-calling capable model)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --enable-auto-tool-choice --tool-call-parser hermes

# 2. install deps
pip install -r requirements.txt

# 3. run the agent
python -m agent.main \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-Coder-32B-Instruct
```

Type a request (e.g. *"write a CUDA kernel for SAXPY and benchmark it"*),
the agent will iterate: think -> call tools -> observe -> repeat,
until it returns a final answer.
