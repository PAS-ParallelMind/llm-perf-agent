"""RunConfig — schema for `agent.batch --config run.yaml`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 2048
    reasoning: bool = False


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
        return cls(
            model=ModelConfig(**raw["model"]),
            agent=AgentSettings(**raw.get("agent", {})),
            io=IOConfig(**raw["io"]),
            system_prompt=raw.get("system_prompt"),
            system_prompt_file=raw.get("system_prompt_file"),
        )
