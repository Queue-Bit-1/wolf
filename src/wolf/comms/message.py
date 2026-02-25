"""Message data structure for the communication layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """An immutable message sent through a communication channel."""

    sender_id: str
    channel: str  # "public", "wolf", "dm:{player1}:{player2}"
    content: str
    day: int = 0
    phase_name: str = ""
    visible_to: frozenset[str] = field(default_factory=frozenset)  # empty = visible to all in channel
    metadata: dict[str, Any] = field(default_factory=dict)
