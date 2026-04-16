"""YAML-based agent configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass
class AgentConfig:
    max_steps: int = 15
    time_budget: int = 300
    workers: int = 1


@dataclass
class EvalConfig:
    launch_configs: str | None = None   # None = use benchmark default
    build_timeout: int = 30
    run_timeout: int = 120


@dataclass
class RunConfig:
    """Top-level config loaded from YAML."""

    model: ModelConfig
    agent: AgentConfig = field(default_factory=AgentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        model = ModelConfig(**raw["model"])
        agent = AgentConfig(**raw.get("agent", {}))
        eval_ = EvalConfig(**raw.get("eval", {}))
        return cls(model=model, agent=agent, eval=eval_)
