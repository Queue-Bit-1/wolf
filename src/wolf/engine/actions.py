"""Action types that agents can submit to the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """Base action submitted by an agent."""

    player_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeakAction(Action):
    """A player speaks during day discussion."""

    content: str = ""


@dataclass(frozen=True)
class VoteAction(Action):
    """A player casts a vote during day voting."""

    target_id: str | None = None  # None = abstain


@dataclass(frozen=True)
class UseAbilityAction(Action):
    """A player uses their role ability (typically at night)."""

    ability_name: str = ""
    target_id: str = ""


@dataclass(frozen=True)
class NoAction(Action):
    """Player chose not to act (or timed out)."""

    reason: str = "no_action"
