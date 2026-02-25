"""Immutable game state and player-filtered view."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from wolf.engine.phase import Phase

if TYPE_CHECKING:
    from wolf.engine.actions import Action
    from wolf.engine.events import GameEvent


@dataclass(frozen=True)
class PlayerSlot:
    """Immutable record for one player in the game."""

    player_id: str
    name: str
    role: str
    team: str
    is_alive: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of the entire game state.

    Every mutation method returns a *new* GameState; the original is never
    modified.
    """

    day: int = 0
    phase: Phase = Phase.SETUP
    players: tuple[PlayerSlot, ...] = ()
    events: tuple[GameEvent, ...] = ()
    night_actions: dict[str, Action] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_player(self, player_id: str) -> PlayerSlot | None:
        """Return the PlayerSlot with the given id, or None."""
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None

    def get_alive_players(self) -> list[PlayerSlot]:
        """Return all living players."""
        return [p for p in self.players if p.is_alive]

    def get_alive_player_ids(self) -> list[str]:
        """Return ids of all living players."""
        return [p.player_id for p in self.players if p.is_alive]

    def get_players_by_role(self, role: str) -> list[PlayerSlot]:
        """Return all players (alive or dead) with *role*."""
        return [p for p in self.players if p.role == role]

    def get_players_by_team(self, team: str) -> list[PlayerSlot]:
        """Return all players (alive or dead) on *team*."""
        return [p for p in self.players if p.team == team]

    # ------------------------------------------------------------------
    # Immutable updates
    # ------------------------------------------------------------------

    def with_phase(self, phase: Phase) -> GameState:
        """Return a copy with a new phase."""
        return replace(self, phase=phase)

    def with_day(self, day: int) -> GameState:
        """Return a copy with a new day counter."""
        return replace(self, day=day)

    def with_player_killed(self, player_id: str) -> GameState:
        """Return a copy where the given player is marked dead."""
        new_players = tuple(
            replace(p, is_alive=False) if p.player_id == player_id else p
            for p in self.players
        )
        return replace(self, players=new_players)

    def with_night_action(self, player_id: str, action: Action) -> GameState:
        """Return a copy with an additional night action recorded."""
        new_actions = dict(self.night_actions)
        new_actions[player_id] = action
        return replace(self, night_actions=new_actions)

    def with_event(self, event: GameEvent) -> GameState:
        """Return a copy with *event* appended to the event log."""
        return replace(self, events=self.events + (event,))

    def clear_night_actions(self) -> GameState:
        """Return a copy with an empty night_actions dict."""
        return replace(self, night_actions={})


class GameStateView:
    """A filtered, read-only view of the game state for a specific player.

    This hides role information of other players so that an agent cannot
    cheat by inspecting the full state.
    """

    def __init__(self, game_state: GameState, player_id: str) -> None:
        self._state = game_state
        self._player_id = player_id

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def day(self) -> int:
        return self._state.day

    @property
    def phase(self) -> Phase:
        return self._state.phase

    @property
    def my_player(self) -> PlayerSlot:
        """Full PlayerSlot for the owning player (includes role)."""
        player = self._state.get_player(self._player_id)
        if player is None:
            raise ValueError(f"Player {self._player_id} not found in game state")
        return player

    @property
    def alive_players(self) -> list[PlayerSlot]:
        """Living players visible to this player (role info stripped for others)."""
        return [
            p if p.player_id == self._player_id
            else replace(p, role="unknown", team="unknown", metadata={})
            for p in self._state.get_alive_players()
        ]

    @property
    def all_players(self) -> list[tuple[str, str, bool]]:
        """All players as ``(player_id, name, is_alive)`` -- no role info."""
        return [
            (p.player_id, p.name, p.is_alive)
            for p in self._state.players
        ]

    @property
    def events(self) -> list[GameEvent]:
        """Events visible to this player.

        Filtering rules:
        - ``PrivateRevealEvent`` only visible to its target player.
        - Wolf-channel ``SpeechEvent`` only visible to werewolf-team players.
        - ``EliminationEvent`` is included but role info should not be
          consumed by agents (the ``on_event`` handler strips it).
        """
        from wolf.engine.events import PrivateRevealEvent, SpeechEvent

        my_player = self._state.get_player(self._player_id)
        is_wolf = my_player is not None and my_player.team == "werewolf"

        visible: list[GameEvent] = []
        for event in self._state.events:
            if isinstance(event, PrivateRevealEvent):
                if event.player_id == self._player_id:
                    visible.append(event)
            elif isinstance(event, SpeechEvent) and event.channel == "wolf":
                if is_wolf:
                    visible.append(event)
            else:
                visible.append(event)
        return visible
