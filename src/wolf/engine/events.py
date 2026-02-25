"""Game event hierarchy for pub/sub observation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wolf.engine.phase import Phase


@dataclass(frozen=True)
class GameEvent:
    """Base game event."""

    day: int = 0
    phase: Phase = Phase.SETUP
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseChangeEvent(GameEvent):
    """The game phase changed."""

    old_phase: Phase = Phase.SETUP
    new_phase: Phase = Phase.SETUP


@dataclass(frozen=True)
class SpeechEvent(GameEvent):
    """A player spoke during discussion."""

    player_id: str = ""
    content: str = ""
    channel: str = "public"


@dataclass(frozen=True)
class VoteEvent(GameEvent):
    """A player cast a vote."""

    voter_id: str = ""
    target_id: str | None = None


@dataclass(frozen=True)
class VoteResultEvent(GameEvent):
    """Result of a vote tally."""

    tally: dict[str, int] = field(default_factory=dict)
    eliminated_id: str | None = None
    tie: bool = False


@dataclass(frozen=True)
class EliminationEvent(GameEvent):
    """A player was eliminated."""

    player_id: str = ""
    role: str = ""
    cause: str = ""  # "vote", "wolf_kill", etc.


@dataclass(frozen=True)
class AbilityUseEvent(GameEvent):
    """A player used a role ability."""

    player_id: str = ""
    ability: str = ""
    target_id: str = ""


@dataclass(frozen=True)
class PrivateRevealEvent(GameEvent):
    """Private information revealed to a player (e.g., seer investigation)."""

    player_id: str = ""
    info: str = ""


@dataclass(frozen=True)
class NightResultEvent(GameEvent):
    """Summary of night actions after resolution."""

    kills: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    saved: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GameEndEvent(GameEvent):
    """The game ended."""

    winning_team: str = ""
    winners: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class ReasoningEvent(GameEvent):
    """Captured reasoning from an LLM agent (for metrics)."""

    player_id: str = ""
    reasoning: str = ""
    action_type: str = ""
