"""Metrics collector that listens to game events and builds summaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from wolf.engine.events import (
    AbilityUseEvent,
    EliminationEvent,
    GameEndEvent,
    GameEvent,
    PhaseChangeEvent,
    ReasoningEvent,
    SpeechEvent,
    VoteEvent,
    VoteResultEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Accumulated statistics for a single player."""

    player_id: str = ""
    name: str = ""
    role: str = ""
    team: str = ""
    speeches: int = 0
    votes_cast: int = 0
    votes_received: int = 0
    abilities_used: int = 0
    survived_until: int = 0
    is_alive: bool = True
    elimination_cause: str = ""
    speech_contents: list[str] = field(default_factory=list)
    vote_targets: list[str | None] = field(default_factory=list)
    ability_targets: list[dict[str, str]] = field(default_factory=list)
    reasoning_log: list[dict[str, str]] = field(default_factory=list)


class MetricsCollector:
    """Event listener that collects per-player and game-wide metrics.

    Implements the event listener interface expected by the engine:
    ``__call__(self, event: GameEvent)``.
    """

    def __init__(self) -> None:
        self._events: list[GameEvent] = []
        self._player_stats: dict[str, PlayerStats] = {}
        self._current_day: int = 0
        self._end_event: GameEndEvent | None = None

    # ------------------------------------------------------------------
    # Event listener interface
    # ------------------------------------------------------------------

    def __call__(self, event: GameEvent) -> None:
        """Process a game event and update internal statistics."""
        self._events.append(event)
        self._current_day = max(self._current_day, event.day)

        if isinstance(event, SpeechEvent):
            self._handle_speech(event)
        elif isinstance(event, VoteEvent):
            self._handle_vote(event)
        elif isinstance(event, EliminationEvent):
            self._handle_elimination(event)
        elif isinstance(event, AbilityUseEvent):
            self._handle_ability_use(event)
        elif isinstance(event, ReasoningEvent):
            self._handle_reasoning(event)
        elif isinstance(event, GameEndEvent):
            self._end_event = event

    # ------------------------------------------------------------------
    # Player registration
    # ------------------------------------------------------------------

    def register_player(
        self, player_id: str, name: str, role: str, team: str
    ) -> None:
        """Register a player so stats can be tracked from the start."""
        self._player_stats[player_id] = PlayerStats(
            player_id=player_id,
            name=name,
            role=role,
            team=team,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _ensure_player(self, player_id: str) -> PlayerStats:
        """Return the PlayerStats for *player_id*, creating if needed."""
        if player_id not in self._player_stats:
            self._player_stats[player_id] = PlayerStats(player_id=player_id)
        return self._player_stats[player_id]

    def _handle_speech(self, event: SpeechEvent) -> None:
        stats = self._ensure_player(event.player_id)
        stats.speeches += 1
        stats.speech_contents.append(event.content)

    def _handle_vote(self, event: VoteEvent) -> None:
        voter = self._ensure_player(event.voter_id)
        voter.votes_cast += 1
        voter.vote_targets.append(event.target_id)

        if event.target_id is not None:
            target = self._ensure_player(event.target_id)
            target.votes_received += 1

    def _handle_elimination(self, event: EliminationEvent) -> None:
        stats = self._ensure_player(event.player_id)
        stats.is_alive = False
        stats.survived_until = event.day
        stats.elimination_cause = event.cause
        # Backfill role/team if we didn't have it yet
        if event.role and not stats.role:
            stats.role = event.role

    def _handle_ability_use(self, event: AbilityUseEvent) -> None:
        stats = self._ensure_player(event.player_id)
        stats.abilities_used += 1
        stats.ability_targets.append(
            {"ability": event.ability, "target": event.target_id}
        )

    def _handle_reasoning(self, event: ReasoningEvent) -> None:
        stats = self._ensure_player(event.player_id)
        stats.reasoning_log.append(
            {"action_type": event.action_type, "reasoning": event.reasoning}
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_game_summary(self) -> dict[str, Any]:
        """Return a structured summary of the entire game.

        Returns
        -------
        dict
            Keys: ``players``, ``events``, ``result``, ``total_days``.
        """
        # Finalize survived_until for players still alive at end of game
        for stats in self._player_stats.values():
            if stats.is_alive:
                stats.survived_until = self._current_day

        players_summary: list[dict[str, Any]] = []
        for stats in self._player_stats.values():
            players_summary.append(
                {
                    "player_id": stats.player_id,
                    "name": stats.name,
                    "role": stats.role,
                    "team": stats.team,
                    "speeches": stats.speeches,
                    "votes_cast": stats.votes_cast,
                    "votes_received": stats.votes_received,
                    "abilities_used": stats.abilities_used,
                    "survived_until": stats.survived_until,
                    "is_alive": stats.is_alive,
                    "elimination_cause": stats.elimination_cause,
                    "speech_contents": stats.speech_contents,
                    "vote_targets": stats.vote_targets,
                    "ability_targets": stats.ability_targets,
                    "reasoning_log": stats.reasoning_log,
                }
            )

        result: dict[str, Any] = {}
        if self._end_event is not None:
            result = {
                "winning_team": self._end_event.winning_team,
                "winners": list(self._end_event.winners),
                "reason": self._end_event.reason,
            }

        return {
            "players": players_summary,
            "result": result,
            "total_days": self._current_day,
            "total_events": len(self._events),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all collected data for a new game."""
        self._events.clear()
        self._player_stats.clear()
        self._current_day = 0
        self._end_event = None
