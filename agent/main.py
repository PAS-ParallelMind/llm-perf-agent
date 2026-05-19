"""Interactive chat REPL — LLM inference perf agent.

Two ways to launch:

  uv run python -m agent.main --config path/to/chat.yaml
  uv run python -m agent.main --model openai/Qwen3-Coder-30B-A3B-Instruct \\
                              --base-url http://localhost:8001/v1

Slash commands inside the REPL:
  /exit, /quit            leave
  /reset                  clear conversation history (keep the session dir)
  /tools                  list registered tools
  /help                   show commands
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel

from . import memory  # noqa: F401  registers remember/recall tools
from .config import AgentSettings, ChatConfig, ModelConfig, SessionConfig
from .engine import Engine
from .fake_engine import FakeEngine
from .loop import ChatAgent, TurnResult
from .tools import TOOLS  # noqa: F401  triggers tool registration
from .types import SessionMeta
from .workspace import set_root

console = Console()


# ---------------------------------------------------------------------------
# Config construction (from CLI flags when no YAML is given)
# ---------------------------------------------------------------------------

def _cfg_from_args(args: argparse.Namespace) -> ChatConfig:
    if args.config:
        return ChatConfig.from_yaml(args.config)
    if not args.dry_run and not args.model:
        raise SystemExit("--model is required unless --config or --dry-run is set")
    return ChatConfig(
        agent=AgentSettings(
            model=ModelConfig(
                name=args.model or "fake-dry-run",
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                reasoning=args.reasoning,
            ),
            max_steps=args.max_steps,
        ),
        session=SessionConfig(dir=args.session_dir, name=args.session_name),
        system_prompt=(Path(args.system_prompt_file).read_text()
                       if args.system_prompt_file else None),
    )


# ---------------------------------------------------------------------------
# Session bookkeeping
# ---------------------------------------------------------------------------

def _init_session(cfg: ChatConfig) -> tuple[Path, Path, SessionMeta]:
    """Create the session dirs, drop a run.yaml snapshot, and return
    (session_root, workspace_dir, meta).

    Layout (webui-compatible):
        <session_dir>/<name>/
          run.yaml
          batch/session/         <- workspace + trace
    """
    name = cfg.session.name or datetime.now().strftime("chat-%Y%m%d-%H%M%S")
    root = Path(cfg.session.dir).expanduser() / name
    workspace = root / "batch" / "session"
    workspace.mkdir(parents=True, exist_ok=True)

    # run.yaml snapshot — so the webui's run-listing logic finds this dir.
    snapshot = {
        "model": {
            "name":     cfg.agent.model.name,
            "base_url": cfg.agent.model.base_url,
        },
        "agent": {"max_steps": cfg.agent.max_steps, "workers": 1},
        "session": {"dir": cfg.session.dir, "name": name},
    }
    (root / "run.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))

    meta = SessionMeta(
        name=name,
        started_at=datetime.now(tz=timezone.utc).isoformat(),
        agent_model=cfg.agent.model.name,
    )
    return root, workspace, meta


def _export_trace(agent: ChatAgent, workspace: Path, meta: SessionMeta) -> None:
    """Re-serialize the full session trace after each turn. Crash-safe."""
    try:
        (workspace / "trace.json").write_text(
            json.dumps(agent.messages, indent=2, default=str)
        )
    except Exception as e:
        console.print(f"[red]trace export failed: {e}[/]")

    try:
        with (workspace / "tool_calls.jsonl").open("w") as f:
            for tc in agent.tool_call_log:
                f.write(json.dumps(tc, default=str) + "\n")
    except Exception as e:
        console.print(f"[red]tool_calls export failed: {e}[/]")

    try:
        summary: dict[str, Any] = {
            "name":            meta.name,
            "started_at":      meta.started_at,
            "agent_model":     meta.agent_model,
            "turns":           meta.turns,
            "total_steps":     meta.total_steps,
            "total_elapsed_s": round(meta.total_elapsed_s, 2),
        }
        (workspace / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception as e:
        console.print(f"[red]summary export failed: {e}[/]")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_HELP = (
    "/exit, /quit   leave the REPL\n"
    "/reset         clear conversation history\n"
    "/tools         list registered tools\n"
    "/help          show this message"
)


def _handle_slash(cmd: str, agent: ChatAgent) -> bool:
    """Return True if the input was a slash command (and was handled)."""
    cmd = cmd.strip()
    if cmd in {"/exit", "/quit"}:
        raise SystemExit(0)
    if cmd == "/help":
        console.print(Panel(_HELP, border_style="yellow", title="commands"))
        return True
    if cmd == "/reset":
        agent.reset()
        console.print("[yellow]conversation reset.[/]")
        return True
    if cmd == "/tools":
        lines = [f"  {n}" for n in sorted(TOOLS)]
        console.print(Panel("\n".join(lines) or "(none)",
                            border_style="cyan", title="tools"))
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM inference perf chat agent")
    ap.add_argument("--config", default=None, help="Path to chat.yaml (ChatConfig)")
    # Fallback CLI flags when --config is not provided
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--system-prompt-file", default=None)
    ap.add_argument("--session-dir", default="runs/chat")
    ap.add_argument("--session-name", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Use FakeEngine — no vLLM required")
    args = ap.parse_args()

    cfg = _cfg_from_args(args)
    root, workspace, meta = _init_session(cfg)
    set_root(workspace)

    if args.dry_run or cfg.agent.model.name == "fake-dry-run":
        engine: Any = FakeEngine()
    else:
        m = cfg.agent.model
        engine = Engine(
            model=m.name,
            base_url=m.base_url,
            api_key=m.api_key,
            temperature=m.temperature,
            max_tokens=m.max_tokens,
            reasoning=m.reasoning,
            timeout=m.timeout,
            max_retries=m.max_retries,
        )

    agent = ChatAgent(
        engine=engine,
        max_steps=cfg.agent.max_steps,
        system_prompt=cfg.resolved_system_prompt(),
    )

    console.print(
        Panel.fit(
            f"[bold]LLM inference perf agent[/]\n"
            f"model:     [cyan]{engine.model}[/]\n"
            f"tools:     [cyan]{', '.join(sorted(TOOLS))}[/]\n"
            f"session:   [cyan]{root}[/]\n"
            f"[dim]/help for commands · Ctrl-D to exit[/]",
            border_style="green",
        )
    )

    while True:
        try:
            user = console.input("\n[bold green]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            break
        if not user:
            continue
        if user.startswith("/") and _handle_slash(user, agent):
            continue

        try:
            result: TurnResult = agent.chat(user)
        except Exception as e:
            console.print(f"[red]error:[/] {e}")
            continue

        meta.turns += 1
        meta.total_steps += result.steps
        meta.total_elapsed_s += result.elapsed_s
        _export_trace(agent, workspace, meta)

        title = (f"assistant · turn {meta.turns} · {result.steps} step(s) · "
                 f"{result.elapsed_s}s")
        console.print(Panel(result.reply or "(no reply)",
                            border_style="cyan", title=title))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
