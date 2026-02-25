"""Pydantic models for all configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for an LLM model endpoint."""

    api_base: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwq:latest"
    reasoning_temperature: float = 0.7
    action_temperature: float = 0.3
    timeout: float = 120.0
    max_tokens: int = 2048
    context_length: int = 16384
    extra_params: dict[str, Any] = Field(default_factory=dict)


class PlayerConfig(BaseModel):
    """Configuration for a single player slot."""

    name: str
    agent_type: str = "llm"
    model: ModelConfig | None = None
    personality: str | None = None


class RoleSlot(BaseModel):
    """A role and its count in the game setup."""

    role: str
    count: int = 1


class VotingConfig(BaseModel):
    """Voting rules configuration."""

    method: str = "plurality"
    allow_no_vote: bool = False
    reveal_votes: bool = True
    tie_breaker: str = "no_elimination"


class CommunicationConfig(BaseModel):
    """Communication channel configuration."""

    allow_wolf_chat: bool = True
    allow_dms: bool = False
    discussion_rounds: int = 2
    max_speech_length: int = 500


class MetricsConfig(BaseModel):
    """Metrics collection configuration."""

    enabled: bool = True
    use_llm_judge: bool = False
    judge_model: ModelConfig | None = None
    output_dir: str = "./results"
    export_formats: list[str] = Field(default_factory=lambda: ["json"])


class BenchmarkConfig(BaseModel):
    """Benchmark suite configuration."""

    num_games: int = 10
    parallel_games: int = 1
    rotate_roles: bool = True
    seed: int | None = None


class GameConfig(BaseModel):
    """Top-level game configuration."""

    game_name: str = "classic_7p"
    num_players: int = 7
    randomize_names: bool = True
    roles: list[RoleSlot] = Field(
        default_factory=lambda: [
            RoleSlot(role="werewolf", count=2),
            RoleSlot(role="seer", count=1),
            RoleSlot(role="doctor", count=1),
            RoleSlot(role="villager", count=3),
        ]
    )
    default_model: ModelConfig = Field(default_factory=ModelConfig)
    model_pool: list[ModelConfig] = Field(default_factory=list)
    players: list[PlayerConfig] = Field(default_factory=list)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    max_days: int = 15
