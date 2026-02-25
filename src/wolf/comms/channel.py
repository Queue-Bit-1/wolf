"""Communication channel abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from wolf.engine.phase import Phase


class Channel(ABC):
    """Abstract base for a communication channel."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique channel identifier."""
        ...

    @abstractmethod
    def can_send(self, player_id: str, phase: Phase) -> bool:
        """Return True if *player_id* may send a message during *phase*."""
        ...

    @abstractmethod
    def can_read(self, player_id: str, phase: Phase) -> bool:
        """Return True if *player_id* may read messages during *phase*."""
        ...

    @property
    @abstractmethod
    def members(self) -> frozenset[str]:
        """Set of player ids that belong to this channel."""
        ...


class PublicChannel(Channel):
    """Town-square channel visible to everyone, writable during day discussion."""

    def __init__(self, player_ids: list[str]) -> None:
        self._player_ids = frozenset(player_ids)

    @property
    def name(self) -> str:
        return "public"

    def can_send(self, player_id: str, phase: Phase) -> bool:
        return player_id in self._player_ids and phase == Phase.DAY_DISCUSSION

    def can_read(self, player_id: str, phase: Phase) -> bool:
        return player_id in self._player_ids

    @property
    def members(self) -> frozenset[str]:
        return self._player_ids


class WolfChannel(Channel):
    """Private wolf channel, writable only at night."""

    def __init__(self, wolf_ids: list[str]) -> None:
        self._wolf_ids = frozenset(wolf_ids)

    @property
    def name(self) -> str:
        return "wolf"

    def can_send(self, player_id: str, phase: Phase) -> bool:
        return player_id in self._wolf_ids and phase == Phase.NIGHT

    def can_read(self, player_id: str, phase: Phase) -> bool:
        return player_id in self._wolf_ids

    @property
    def members(self) -> frozenset[str]:
        return self._wolf_ids


class DirectMessageChannel(Channel):
    """Private channel between exactly two players."""

    def __init__(
        self,
        player1: str,
        player2: str,
        allowed_phases: list[Phase] | None = None,
    ) -> None:
        self._members = frozenset({player1, player2})
        self._player1 = player1
        self._player2 = player2
        # Default: allow DMs during DAY_DISCUSSION only.
        self._allowed_phases: frozenset[Phase] = (
            frozenset(allowed_phases) if allowed_phases else frozenset({Phase.DAY_DISCUSSION})
        )

    @property
    def name(self) -> str:
        # Canonical ordering so the name is deterministic regardless of
        # which player was passed first.
        ids = sorted(self._members)
        return f"dm:{ids[0]}:{ids[1]}"

    def can_send(self, player_id: str, phase: Phase) -> bool:
        return player_id in self._members and phase in self._allowed_phases

    def can_read(self, player_id: str, phase: Phase) -> bool:
        return player_id in self._members

    @property
    def members(self) -> frozenset[str]:
        return self._members
