"""Live chat-session management for the webui backend.

Keeps a process-local registry of :class:`agent.loop.ChatAgent` instances
keyed by session name and persists each turn's trace to the same on-disk
layout the CLI uses, so the existing trace endpoints continue to work.

Sessions can be created via :func:`create_session` (POST /api/sessions) or
resumed lazily from disk when a chat request arrives for a session that
hasn't been loaded yet — the resume path replays the saved ``trace.json``
into a fresh :class:`ChatAgent` so the conversation continues coherently.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.config import AgentSettings, ChatConfig, ModelConfig, SessionConfig
from agent.engine import Engine
from agent.fake_engine import FakeEngine
from agent.loop import ChatAgent, TurnResult
from agent.tools import TOOLS  # noqa: F401 — triggers tool registration
from agent.types import SessionMeta
from agent.workspace import set_root

_SESSION_WORKSPACE = "session"


# ---------------------------------------------------------------------------
# Disk layout helpers
# ---------------------------------------------------------------------------

def _workspace(root: Path) -> Path:
    return root / "batch" / _SESSION_WORKSPACE


def _write_run_yaml(root: Path, cfg: ChatConfig) -> None:
    """Snapshot the full ChatConfig so the session can be resumed later."""
    m = cfg.agent.model
    payload: dict[str, Any] = {
        "agent": {
            "model": {
                "name":              m.name,
                "base_url":          m.base_url,
                "api_key":           m.api_key,
                "temperature":       m.temperature,
                "max_output_tokens": m.max_output_tokens,
                "max_model_len":     m.max_model_len,
                "reasoning":         m.reasoning,
                "timeout":           m.timeout,
                "max_retries":       m.max_retries,
            },
            "max_steps": cfg.agent.max_steps,
        },
        "session": asdict(cfg.session),
    }
    if cfg.system_prompt is not None:
        payload["system_prompt"] = cfg.system_prompt
    if cfg.system_prompt_file is not None:
        payload["system_prompt_file"] = cfg.system_prompt_file
    (root / "run.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def _export_trace(agent: ChatAgent, workspace: Path, meta: SessionMeta) -> None:
    """Mirror agent.main._export_trace — same on-disk format the trace
    endpoints already serve."""
    (workspace / "trace.json").write_text(
        json.dumps(agent.messages, indent=2, default=str)
    )
    with (workspace / "tool_calls.jsonl").open("w") as f:
        for tc in agent.tool_call_log:
            f.write(json.dumps(tc, default=str) + "\n")
    with (workspace / "llm_requests.jsonl").open("w") as f:
        for req in agent.llm_requests:
            f.write(json.dumps(req, default=str) + "\n")
    (workspace / "summary.json").write_text(json.dumps({
        "name":            meta.name,
        "started_at":      meta.started_at,
        "agent_model":     meta.agent_model,
        "turns":           meta.turns,
        "total_steps":     meta.total_steps,
        "total_elapsed_s": round(meta.total_elapsed_s, 2),
    }, indent=2))


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

def _build_engine(model: ModelConfig) -> Any:
    """Return a live Engine, or a FakeEngine when the config marks dry-run."""
    if model.name == "fake-dry-run":
        return FakeEngine()
    return Engine(
        model=model.name,
        base_url=model.base_url,
        api_key=model.api_key,
        temperature=model.temperature,
        max_output_tokens=model.max_output_tokens,
        max_model_len=model.max_model_len,
        reasoning=model.reasoning,
        timeout=model.timeout,
        max_retries=model.max_retries,
    )


# ---------------------------------------------------------------------------
# In-memory session registry
# ---------------------------------------------------------------------------

class _SessionEntry:
    __slots__ = ("agent", "meta", "workspace", "root", "lock")

    def __init__(self, agent: ChatAgent, meta: SessionMeta,
                 workspace: Path, root: Path) -> None:
        self.agent = agent
        self.meta = meta
        self.workspace = workspace
        self.root = root
        # Serializes concurrent chat() calls against the same session.
        self.lock = threading.Lock()


class SessionStore:
    """Process-local registry. Sessions are created (or resumed from disk)
    on demand and kept alive until the server restarts."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._entries: dict[str, _SessionEntry] = {}
        self._registry_lock = threading.Lock()

    # -- creation -----------------------------------------------------------

    def create(self, cfg: ChatConfig) -> str:
        name = cfg.session.name or datetime.now().strftime("chat-%Y%m%d-%H%M%S")
        if cfg.session.dir != str(self.runs_dir):
            # The webui only manages sessions under its configured runs
            # dir; honoring an arbitrary dir here would leak files outside
            # the listing endpoint's view.
            cfg.session.dir = str(self.runs_dir)
        cfg.session.name = name

        root = self.runs_dir / name
        if root.exists():
            raise ValueError(f"session already exists: {name}")
        ws = _workspace(root)
        ws.mkdir(parents=True, exist_ok=True)

        _write_run_yaml(root, cfg)
        # The tools that write to the workspace (fs, bash, modeling) anchor
        # on agent.workspace.ROOT. set_root is process-global, so the most
        # recently used session wins — fine for a single-user dev UI.
        set_root(ws)

        engine = _build_engine(cfg.agent.model)
        agent = ChatAgent(
            engine=engine,
            max_steps=cfg.agent.max_steps,
            system_prompt=cfg.resolved_system_prompt(),
        )
        meta = SessionMeta(
            name=name,
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            agent_model=cfg.agent.model.name,
        )
        entry = _SessionEntry(agent=agent, meta=meta, workspace=ws, root=root)
        # Persist an empty trace + summary so the listing endpoint shows
        # the session immediately, even before the first turn.
        _export_trace(agent, ws, meta)

        with self._registry_lock:
            self._entries[name] = entry
        return name

    # -- chat ---------------------------------------------------------------

    def chat(self, name: str, message: str) -> dict[str, Any]:
        entry = self._get_or_resume(name)
        with entry.lock:
            # Re-point the workspace anchor at this session's dir in case
            # another session was used in between.
            set_root(entry.workspace)
            result: TurnResult = entry.agent.chat(message)
            entry.meta.turns += 1
            entry.meta.total_steps += result.steps
            entry.meta.total_elapsed_s += result.elapsed_s
            _export_trace(entry.agent, entry.workspace, entry.meta)
            return {
                "reply":      result.reply,
                "steps":      result.steps,
                "elapsed_s":  result.elapsed_s,
                "truncated":  result.truncated,
                "messages":   list(entry.agent.messages),
                "turns":      entry.meta.turns,
            }

    # -- maintenance --------------------------------------------------------

    def reset(self, name: str) -> None:
        entry = self._get_or_resume(name)
        with entry.lock:
            entry.agent.reset()
            entry.meta.turns = 0
            entry.meta.total_steps = 0
            entry.meta.total_elapsed_s = 0.0
            _export_trace(entry.agent, entry.workspace, entry.meta)

    # -- resume from disk ---------------------------------------------------

    def _get_or_resume(self, name: str) -> _SessionEntry:
        with self._registry_lock:
            if name in self._entries:
                return self._entries[name]

        root = (self.runs_dir / name).resolve()
        if not root.is_dir() or self.runs_dir not in root.parents:
            raise FileNotFoundError(f"no such session: {name}")
        run_yaml = root / "run.yaml"
        if not run_yaml.is_file():
            raise FileNotFoundError(f"session has no run.yaml: {name}")

        cfg = ChatConfig.from_yaml(run_yaml)
        ws = _workspace(root)
        ws.mkdir(parents=True, exist_ok=True)
        set_root(ws)

        engine = _build_engine(cfg.agent.model)
        agent = ChatAgent(
            engine=engine,
            max_steps=cfg.agent.max_steps,
            system_prompt=cfg.resolved_system_prompt(),
        )
        # Replay the saved trace so the conversation continues. The system
        # message ChatAgent's __init__ added is replaced by the first
        # saved message (which was its own system message). If there's no
        # saved trace yet, keep the fresh init.
        trace_path = ws / "trace.json"
        if trace_path.is_file():
            try:
                saved = json.loads(trace_path.read_text())
                if isinstance(saved, list) and saved:
                    agent.messages = saved
                    agent.turn_count = sum(1 for m in saved if m.get("role") == "user")
            except (OSError, json.JSONDecodeError):
                pass

        # Pull existing counters from summary.json so totals don't reset.
        meta = SessionMeta(
            name=name,
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            agent_model=cfg.agent.model.name,
        )
        summary_path = ws / "summary.json"
        if summary_path.is_file():
            try:
                s = json.loads(summary_path.read_text())
                meta.started_at      = s.get("started_at", meta.started_at)
                meta.turns           = int(s.get("turns") or 0)
                meta.total_steps     = int(s.get("total_steps") or 0)
                meta.total_elapsed_s = float(s.get("total_elapsed_s") or 0.0)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        entry = _SessionEntry(agent=agent, meta=meta, workspace=ws, root=root)
        with self._registry_lock:
            # Race: another thread may have resumed it; prefer that one.
            self._entries.setdefault(name, entry)
            return self._entries[name]


# ---------------------------------------------------------------------------
# Preset / default-config discovery
# ---------------------------------------------------------------------------

def discover_presets(harness_root: Path) -> list[dict[str, Any]]:
    """Return one entry per ``*.yaml`` in the harness root that looks like
    a ChatConfig (has ``agent.model.name``). The user picks one of these in
    the new-session UI; the form pre-fills from the picked preset.

    Also synthesises a built-in ``fake-dry-run`` preset so the UI can
    spin up a session without a vLLM server."""
    presets: list[dict[str, Any]] = [{
        "file":  "<builtin>",
        "label": "fake-dry-run (no server)",
        "agent": {
            "model": {
                "name":              "fake-dry-run",
                "base_url":          "",
                "api_key":           "EMPTY",
                "temperature":       0.0,
                "max_output_tokens": 2048,
                "max_model_len":     32768,
                "reasoning":         False,
            },
            "max_steps": 8,
        },
    }]
    for p in sorted(harness_root.glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        model = ((raw.get("agent") or {}).get("model") or {})
        if not model.get("name"):
            continue
        presets.append({
            "file":  p.name,
            "label": p.stem,
            "agent": raw.get("agent") or {},
            "system_prompt":      raw.get("system_prompt"),
            "system_prompt_file": raw.get("system_prompt_file"),
        })
    return presets


def cfg_from_form(form: dict[str, Any], runs_dir: Path) -> ChatConfig:
    """Build a ChatConfig from a JSON payload posted by the new-session UI.

    Only the fields the form actually surfaces are honored; the rest fall
    back to ModelConfig / AgentSettings defaults."""
    model_raw = dict(form.get("model") or {})
    if not model_raw.get("name"):
        raise ValueError("model.name is required")
    # Drop empty-string fields so dataclass defaults kick in.
    model_raw = {k: v for k, v in model_raw.items() if v not in ("", None)}
    model = ModelConfig(**model_raw)

    agent = AgentSettings(
        model=model,
        max_steps=int(form.get("max_steps") or 20),
    )
    session = SessionConfig(
        dir=str(runs_dir),
        name=form.get("name") or None,
    )
    return ChatConfig(
        agent=agent,
        session=session,
        system_prompt=form.get("system_prompt"),
        system_prompt_file=form.get("system_prompt_file"),
    )
