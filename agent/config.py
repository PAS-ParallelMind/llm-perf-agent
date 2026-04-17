"""YAML-based configuration: agent config + benchmark config."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Agent config (shared across all benchmarks)
# ---------------------------------------------------------------------------

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
class AgentConfig:
    """Loaded from agent.yaml at project root."""

    model: ModelConfig
    agent: AgentSettings = field(default_factory=AgentSettings)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        raw = yaml.safe_load(Path(path).read_text())
        model = ModelConfig(**raw["model"])
        agent = AgentSettings(**raw.get("agent", {}))
        return cls(model=model, agent=agent)


# ---------------------------------------------------------------------------
# Benchmark config (per-run, benchmark-specific)
# ---------------------------------------------------------------------------

@dataclass
class ParevalBenchmarkConfig:
    """Loaded from runs/<run-name>/config.yaml for ParEval runs."""

    problem_set: str = "omp"
    launch_configs: str | None = None
    build_timeout: int = 30
    run_timeout: int = 120

    @classmethod
    def from_yaml(cls, path: str | Path) -> ParevalBenchmarkConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)
