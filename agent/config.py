"""RunConfig — schema for `agent.batch --config run.yaml`."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Expand ${VAR} / $VAR references inside string fields when the value
# is wrapped in that exact form (e.g. ``api_key: ${ANTHROPIC_API_KEY}``).
# Lets the run.yaml keep credentials out of the file.
_ENVVAR_RE = re.compile(r"^\s*\$\{?(\w+)\}?\s*$")


def _expand_env(s: str | None) -> str | None:
    if not isinstance(s, str):
        return s
    m = _ENVVAR_RE.match(s)
    if not m:
        return s
    val = os.environ.get(m.group(1))
    if val is None:
        raise RuntimeError(f"env var {m.group(1)!r} referenced in run.yaml is not set")
    return val


@dataclass
class ModelConfig:
    name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    # Set to None to omit `temperature` from the API call (some
    # Anthropic models reject it).
    temperature: float | None = 0.0
    max_tokens: int = 2048
    reasoning: bool = False
    timeout: float = 900.0
    max_retries: int = 2


@dataclass
class AgentSettings:
    max_steps: int = 15
    time_budget: int = 300
    workers: int = 1


@dataclass
class IOConfig:
    input: str                          # problems.json (list[{id, prompt, ...}])
    output: str                         # results JSON
    workspace_root: str = "runs/batch"  # per-task subdirs created here


@dataclass
class RunConfig:
    model: ModelConfig
    agent: AgentSettings
    io: IOConfig
    # Inline string (preferred) or path to a text file. If both are set,
    # the inline string wins.
    system_prompt: str | None = None
    system_prompt_file: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        model_raw = dict(raw["model"])
        for k in ("api_key", "base_url", "name"):
            if k in model_raw:
                model_raw[k] = _expand_env(model_raw[k])
        return cls(
            model=ModelConfig(**model_raw),
            agent=AgentSettings(**raw.get("agent", {})),
            io=IOConfig(**raw["io"]),
            system_prompt=raw.get("system_prompt"),
            system_prompt_file=raw.get("system_prompt_file"),
        )
