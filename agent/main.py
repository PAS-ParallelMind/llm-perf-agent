"""Interactive CLI entry point."""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from . import memory  # noqa: F401  registers remember/recall tools
from .engine import Engine
from .fake_engine import FakeEngine
from .loop import Agent
from .tools import TOOLS  # noqa: F401  triggers tool registration
from .workspace import DEFAULT_ROOT, set_root

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser(description="cc-style CUDA/MPI/OpenMP agent")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--reasoning", action="store_true",
                    help="Model emits a reasoning trace; echo it back each turn")
    ap.add_argument(
        "--workspace",
        default=str(DEFAULT_ROOT),
        help=f"Sandbox root for all file/build/run tools (default {DEFAULT_ROOT})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a scripted fake engine (no vLLM required) to exercise the loop",
    )
    args = ap.parse_args()

    workspace_root = set_root(args.workspace)

    if args.dry_run:
        engine = FakeEngine()
    else:
        if not args.model:
            ap.error("--model is required unless --dry-run is set")
        engine = Engine(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            reasoning=args.reasoning,
        )
    agent = Agent(engine, max_steps=args.max_steps)

    console.print(
        Panel.fit(
            f"[bold]cc-style parallel-code agent[/]\n"
            f"model: [cyan]{engine.model}[/]  "
            f"tools: [cyan]{len(TOOLS)}[/]\n"
            f"workspace: [cyan]{workspace_root}[/]\n"
            f"type your request (Ctrl-D or 'exit' to quit)",
            border_style="green",
        )
    )

    while True:
        try:
            user = console.input("\n[bold green]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            return
        try:
            result = agent.run(user)
            reply = result.raw_reply
        except Exception as e:
            console.print(f"[red]error:[/] {e}")
            continue
        console.print(Panel(reply or "(no reply)", border_style="cyan", title="assistant"))


if __name__ == "__main__":
    main()
