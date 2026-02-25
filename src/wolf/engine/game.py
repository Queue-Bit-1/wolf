"""Top-level Game class that orchestrates a complete Werewolf game."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from wolf.engine.events import GameEndEvent, PhaseChangeEvent, VoteResultEvent
from wolf.engine.moderator import Moderator
from wolf.engine.phase import Phase
from wolf.engine.state import GameState, GameStateView, PlayerSlot
from wolf.engine.victory import check_victory

if TYPE_CHECKING:
    from wolf.config.schema import GameConfig

logger = logging.getLogger(__name__)


class Game:
    """Orchestrates a single Werewolf game from setup to conclusion.

    Parameters
    ----------
    config:
        Full game configuration (roles, voting rules, communication, etc.).
    agents:
        Mapping of ``player_id`` to agent instances that implement the
        ``AgentBase`` interface.
    role_registry:
        Registry that maps role names to ``RoleBase`` objects.
    channel_manager:
        Communication channel manager for broadcasting messages.
    event_listeners:
        List of callables invoked for every ``GameEvent``.
    """

    def __init__(
        self,
        config: GameConfig,
        agents: dict[str, Any],
        role_registry: Any,
        channel_manager: Any,
        event_listeners: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.agents = agents
        self.role_registry = role_registry
        self.channel_manager = channel_manager
        self.event_listeners: list[Any] = event_listeners or []
        self.state: GameState = GameState()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> GameEndEvent:
        """Execute the full game loop and return the final ``GameEndEvent``."""

        # ---- Setup ----
        self.state = self._setup_players()

        # Emit initial phase-change event.
        setup_event = PhaseChangeEvent(
            day=0,
            phase=Phase.SETUP,
            old_phase=Phase.SETUP,
            new_phase=Phase.NIGHT,
        )
        self.state = self.state.with_event(setup_event)
        self._emit(setup_event)

        # Notify agents that the game has started.
        for player_id, agent in self.agents.items():
            view = GameStateView(self.state, player_id)
            try:
                await agent.on_game_start(view)
            except Exception:
                logger.exception(
                    "Agent %s raised during on_game_start", player_id
                )

        # ---- Main loop ----
        moderator = Moderator(
            state=self.state,
            agents=self.agents,
            channel_manager=self.channel_manager,
            event_listeners=self.event_listeners,
            config=self.config,
            role_registry=self.role_registry,
        )

        day = 0
        while day < self.config.max_days:
            day += 1

            # -- NIGHT --
            self.state = self.state.with_day(day)
            self.state = self._transition(Phase.NIGHT)
            moderator.state = self.state
            self.state = await moderator.run_night(self.state)

            # -- DAWN --
            self.state = self._transition(Phase.DAWN)
            moderator.state = self.state
            self.state = await moderator.run_dawn(self.state)

            # Check victory after dawn (wolf kills may end the game).
            result = check_victory(self.state)
            if result is not None:
                return await self._end_game(result)

            # -- DAY_DISCUSSION --
            self.state = self._transition(Phase.DAY_DISCUSSION)
            moderator.state = self.state
            self.state = await moderator.run_discussion(self.state)

            # -- DAY_VOTE --
            self.state = self._transition(Phase.DAY_VOTE)
            moderator.state = self.state
            self.state = await moderator.run_vote(self.state)

            # -- DAY_VOTE_RESULT --
            self.state = self._transition(Phase.DAY_VOTE_RESULT)
            moderator.state = self.state

            # Find the most recent VoteResultEvent to hand to run_vote_result.
            vote_result_event = self._last_vote_result()
            if vote_result_event is not None:
                self.state = await moderator.run_vote_result(
                    self.state, vote_result_event
                )

            # Check victory after vote result.
            result = check_victory(self.state)
            if result is not None:
                return await self._end_game(result)

        # Max days exceeded -- declare a draw / werewolf win by default.
        return await self._end_game(
            GameEndEvent(
                day=day,
                phase=self.state.phase,
                winning_team="werewolf",
                winners=[
                    p.player_id
                    for p in self.state.get_players_by_team("werewolf")
                ],
                reason=f"Game exceeded the maximum of {self.config.max_days} days.",
            )
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _setup_players(self) -> GameState:
        """Build the initial ``GameState`` with player slots from config.

        Roles are assigned in the order they appear in the config's role
        list, matched against the agents dict.
        """
        roles: list[tuple[str, str]] = []  # (role_name, team)
        for role_slot in self.config.roles:
            role_obj = self.role_registry.get(role_slot.role)
            team = getattr(role_obj, "team", "village") if role_obj else "village"
            for _ in range(role_slot.count):
                roles.append((role_slot.role, team))

        player_ids = list(self.agents.keys())
        player_configs = {pc.name: pc for pc in self.config.players}

        slots: list[PlayerSlot] = []
        for idx, player_id in enumerate(player_ids):
            role_name, team = roles[idx] if idx < len(roles) else ("villager", "village")
            # Try to find a matching PlayerConfig for the name.
            pc = player_configs.get(player_id)
            name = pc.name if pc else player_id
            slots.append(
                PlayerSlot(
                    player_id=player_id,
                    name=name,
                    role=role_name,
                    team=team,
                )
            )

        return GameState(
            day=0,
            phase=Phase.SETUP,
            players=tuple(slots),
        )

    def _transition(self, new_phase: Phase) -> GameState:
        """Transition to *new_phase*, emitting a ``PhaseChangeEvent``."""
        old_phase = self.state.phase
        state = self.state.with_phase(new_phase)

        event = PhaseChangeEvent(
            day=state.day,
            phase=new_phase,
            old_phase=old_phase,
            new_phase=new_phase,
        )
        state = state.with_event(event)
        self._emit(event)
        self.state = state
        return state

    def _emit(self, event: Any) -> None:
        """Dispatch *event* to all registered listeners."""
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("Event listener raised an exception")

    def _last_vote_result(self) -> VoteResultEvent | None:
        """Find the most recent ``VoteResultEvent`` in the state's event log."""
        for event in reversed(self.state.events):
            if isinstance(event, VoteResultEvent):
                return event
        return None

    async def _end_game(self, result: GameEndEvent) -> GameEndEvent:
        """Finalize the game: transition to GAME_OVER, emit event, notify
        agents, and return the result."""
        self.state = self.state.with_phase(Phase.GAME_OVER)

        phase_event = PhaseChangeEvent(
            day=self.state.day,
            phase=Phase.GAME_OVER,
            old_phase=self.state.phase,
            new_phase=Phase.GAME_OVER,
        )
        self.state = self.state.with_event(phase_event)
        self._emit(phase_event)

        self.state = self.state.with_event(result)
        self._emit(result)

        for player_id, agent in self.agents.items():
            try:
                await agent.on_game_end(result)
            except Exception:
                logger.exception(
                    "Agent %s raised during on_game_end", player_id
                )

        return result
