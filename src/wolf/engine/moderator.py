"""Moderator -- drives phase transitions and agent interactions via toolkits."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from wolf.engine.events import (
    EliminationEvent,
    PhaseChangeEvent,
    SpeechEvent,
    VoteEvent,
    VoteResultEvent,
)
from wolf.engine.phase import Phase
from wolf.engine.resolver import resolve_night
from wolf.engine.state import GameState, PlayerSlot

if TYPE_CHECKING:
    from wolf.agents.briefing_builder import BriefingBuilder
    from wolf.agents.knowledge_base import KnowledgeBase
    from wolf.agents.tool_factory import ToolFactory
    from wolf.config.schema import GameConfig
    from wolf.engine.actions import Action
    from wolf.engine.events import GameEvent

logger = logging.getLogger(__name__)


class Moderator:
    """Drives a single game through its phase transitions.

    Uses the tool-based architecture: builds briefings and toolkits for
    each agent per phase, then calls ``agent.run_phase(briefing, toolkit)``.

    All state mutations go through the immutable ``GameState`` -- the
    moderator never mutates state in-place.
    """

    def __init__(
        self,
        state: GameState,
        agents: dict[str, Any],
        channel_manager: Any,
        event_listeners: list[Any],
        config: GameConfig,
        role_registry: Any | None = None,
        tool_factory: ToolFactory | None = None,
        briefing_builder: BriefingBuilder | None = None,
        knowledge_bases: dict[str, KnowledgeBase] | None = None,
    ) -> None:
        self.state = state
        self.agents = agents
        self.channel_manager = channel_manager
        self.event_listeners = event_listeners
        self.config = config
        self.role_registry = role_registry
        self.tool_factory = tool_factory
        self.briefing_builder = briefing_builder
        self.knowledge_bases = knowledge_bases or {}

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def emit_event(self, event: GameEvent) -> None:
        """Notify all registered event listeners about *event*."""
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("Event listener raised an exception")

    # ------------------------------------------------------------------
    # Name helpers
    # ------------------------------------------------------------------

    def _name(self, player_id: str) -> str:
        if self.tool_factory:
            return self.tool_factory._id_to_name.get(player_id, player_id)
        return player_id

    # ------------------------------------------------------------------
    # Phase runners
    # ------------------------------------------------------------------

    async def run_night(self, state: GameState) -> GameState:
        """Solicit night abilities from alive players.

        If wolf chat is enabled and there are 2+ alive wolves, a wolf
        discussion round runs first so wolves can coordinate their kill.
        """
        from wolf.engine.actions import NoAction, UseAbilityAction

        state = state.clear_night_actions()

        # --- Wolf chat: let wolves discuss before kill decisions ---
        wolf_chat_messages: list[str] = []
        if self.config.communication.allow_wolf_chat:
            alive_wolves = [
                p for p in state.get_alive_players() if p.team == "werewolf"
            ]
            if len(alive_wolves) >= 2:
                state, wolf_chat_messages = await self._run_wolf_chat(state, alive_wolves)

        # --- Solicit night abilities from all eligible players ---
        for player in state.get_alive_players():
            if self.role_registry is None:
                continue
            role = self.role_registry.get(player.role)
            if role is None:
                continue

            night_abilities = [
                a for a in getattr(role, "abilities", [])
                if a.phase == Phase.NIGHT
            ]
            if not night_abilities:
                continue

            agent = self.agents.get(player.player_id)
            if agent is None:
                continue

            ability = night_abilities[0]

            # Build briefing and toolkit
            if self.tool_factory and self.briefing_builder:
                self.tool_factory.state = state  # update state reference
                kb = self.knowledge_bases.get(player.player_id)

                allies = None
                wolf_msgs_for_player = None
                if player.team == "werewolf":
                    allies = [
                        self._name(w.player_id)
                        for w in state.get_alive_players()
                        if w.team == "werewolf" and w.player_id != player.player_id
                    ]
                    wolf_msgs_for_player = wolf_chat_messages or None

                briefing = self.briefing_builder.build_night_briefing(
                    state=state,
                    player_id=player.player_id,
                    kb=kb,
                    role_name=role.name,
                    ability_name=ability.name,
                    ability_description=ability.description,
                    allies=allies,
                    wolf_chat_messages=wolf_msgs_for_player,
                )
                toolkit = self.tool_factory.build_night_toolkit(
                    player_id=player.player_id,
                    ability_name=ability.name,
                )

                try:
                    action: Action = await agent.run_phase(briefing, toolkit)
                except Exception:
                    logger.exception(
                        "Agent %s raised during night action", player.player_id
                    )
                    action = NoAction(player_id=player.player_id, reason="error")
            else:
                # Fallback for agents without toolkit support
                action = NoAction(player_id=player.player_id, reason="no_toolkit")

            # Record the night action
            if isinstance(action, UseAbilityAction):
                state = state.with_night_action(player.player_id, action)
            else:
                state = state.with_night_action(
                    player.player_id,
                    NoAction(player_id=player.player_id, reason="no_ability_used"),
                )

        return state

    async def _run_wolf_chat(
        self, state: GameState, alive_wolves: list[PlayerSlot]
    ) -> tuple[GameState, list[str]]:
        """Run a wolf-only discussion round during night.

        Each wolf speaks in sequence on the 'wolf' channel. Prior wolf
        messages are fed to subsequent wolves via briefings.

        Returns the updated state and the list of chat messages so that
        the night ability briefing can include what the pack discussed.
        """
        wolf_names = {w.player_id: self._name(w.player_id) for w in alive_wolves}
        prior_messages: list[str] = []

        for wolf in alive_wolves:
            agent = self.agents.get(wolf.player_id)
            if agent is None:
                continue

            allies = [n for pid, n in wolf_names.items() if pid != wolf.player_id]

            if self.tool_factory and self.briefing_builder:
                self.tool_factory.state = state
                kb = self.knowledge_bases.get(wolf.player_id)

                briefing = self.briefing_builder.build_wolf_chat_briefing(
                    state=state,
                    player_id=wolf.player_id,
                    kb=kb,
                    allies=allies,
                    prior_messages=prior_messages,
                )
                toolkit = self.tool_factory.build_wolf_chat_toolkit(
                    player_id=wolf.player_id,
                )

                try:
                    action = await agent.run_phase(briefing, toolkit)
                except Exception:
                    logger.exception(
                        "Agent %s raised during wolf chat", wolf.player_id
                    )
                    continue

                from wolf.engine.actions import SpeakAction
                if isinstance(action, SpeakAction) and action.content:
                    event = SpeechEvent(
                        day=state.day,
                        phase=Phase.NIGHT,
                        player_id=wolf.player_id,
                        content=action.content,
                        channel="wolf",
                    )
                    state = state.with_event(event)
                    self.emit_event(event)
                    prior_messages.append(f"{wolf_names[wolf.player_id]}: {action.content}")

        return state, prior_messages

    async def run_dawn(self, state: GameState) -> GameState:
        """Resolve all night actions and emit the resulting events."""
        state, events = resolve_night(state, self.role_registry)

        for event in events:
            self.emit_event(event)

        return state

    async def run_discussion(self, state: GameState) -> GameState:
        """Run discussion rounds where each alive player speaks."""
        from wolf.engine.actions import SpeakAction

        rounds = self.config.communication.discussion_rounds
        speeches_so_far: list[tuple[str, str]] = []

        for _round in range(rounds):
            for player in state.get_alive_players():
                if not player.is_alive:
                    continue
                agent = self.agents.get(player.player_id)
                if agent is None:
                    continue

                if self.tool_factory and self.briefing_builder:
                    self.tool_factory.state = state
                    kb = self.knowledge_bases.get(player.player_id)
                    role = self.role_registry.get(player.role) if self.role_registry else None
                    role_name = role.name if role else player.role

                    briefing = self.briefing_builder.build_discussion_briefing(
                        state=state,
                        player_id=player.player_id,
                        kb=kb,
                        role_name=role_name,
                        speeches_so_far=speeches_so_far,
                    )
                    toolkit = self.tool_factory.build_discussion_toolkit(
                        player_id=player.player_id,
                    )

                    try:
                        action = await agent.run_phase(briefing, toolkit)
                    except Exception:
                        logger.exception(
                            "Agent %s raised during discussion", player.player_id
                        )
                        continue
                else:
                    continue

                if isinstance(action, SpeakAction) and action.content:
                    event = SpeechEvent(
                        day=state.day,
                        phase=Phase.DAY_DISCUSSION,
                        player_id=player.player_id,
                        content=action.content,
                        channel="public",
                    )
                    state = state.with_event(event)
                    self.emit_event(event)
                    speeches_so_far.append(
                        (self._name(player.player_id), action.content)
                    )

        return state

    async def run_vote(self, state: GameState) -> GameState:
        """Solicit votes from all alive players, tally, and produce a VoteResultEvent."""
        from wolf.engine.actions import VoteAction

        votes: dict[str, str | None] = {}

        for player in state.get_alive_players():
            agent = self.agents.get(player.player_id)
            if agent is None:
                continue

            if self.tool_factory and self.briefing_builder:
                self.tool_factory.state = state
                kb = self.knowledge_bases.get(player.player_id)
                role = self.role_registry.get(player.role) if self.role_registry else None
                role_name = role.name if role else player.role

                valid_targets = [
                    self._name(p.player_id)
                    for p in state.get_alive_players()
                    if p.player_id != player.player_id
                ]

                briefing = self.briefing_builder.build_vote_briefing(
                    state=state,
                    player_id=player.player_id,
                    kb=kb,
                    role_name=role_name,
                    valid_targets=valid_targets,
                )
                toolkit = self.tool_factory.build_vote_toolkit(
                    player_id=player.player_id,
                )

                try:
                    action = await agent.run_phase(briefing, toolkit)
                except Exception:
                    logger.exception(
                        "Agent %s raised during voting", player.player_id
                    )
                    action = VoteAction(player_id=player.player_id, target_id=None)
            else:
                action = VoteAction(player_id=player.player_id, target_id=None)

            if isinstance(action, VoteAction):
                votes[player.player_id] = action.target_id
            else:
                votes[player.player_id] = None

        # Emit individual vote events
        for voter_id, target_id in votes.items():
            vote_event = VoteEvent(
                day=state.day,
                phase=Phase.DAY_VOTE,
                voter_id=voter_id,
                target_id=target_id,
            )
            state = state.with_event(vote_event)
            self.emit_event(vote_event)

        # Tally
        tally: Counter[str] = Counter()
        for target_id in votes.values():
            if target_id is not None:
                tally[target_id] += 1

        eliminated_id: str | None = None
        tie = False

        if tally:
            max_votes = tally.most_common(1)[0][1]
            top_candidates = [pid for pid, cnt in tally.items() if cnt == max_votes]

            if len(top_candidates) == 1:
                eliminated_id = top_candidates[0]
            else:
                tie = True
                tie_breaker = self.config.voting.tie_breaker
                if tie_breaker == "no_elimination":
                    eliminated_id = None
                elif tie_breaker == "random":
                    import random
                    eliminated_id = random.choice(top_candidates)
                else:
                    eliminated_id = None

        vote_result = VoteResultEvent(
            day=state.day,
            phase=Phase.DAY_VOTE,
            tally=dict(tally),
            eliminated_id=eliminated_id,
            tie=tie,
        )
        state = state.with_event(vote_result)
        self.emit_event(vote_result)

        return state

    async def run_vote_result(
        self, state: GameState, vote_result_event: VoteResultEvent
    ) -> GameState:
        """Apply the elimination determined by the vote, if any."""
        eliminated_id = vote_result_event.eliminated_id

        if eliminated_id is not None:
            target = state.get_player(eliminated_id)
            if target is not None and target.is_alive:
                state = state.with_player_killed(eliminated_id)
                elim_event = EliminationEvent(
                    day=state.day,
                    phase=Phase.DAY_VOTE_RESULT,
                    player_id=eliminated_id,
                    role=target.role,
                    cause="vote",
                )
                state = state.with_event(elim_event)
                self.emit_event(elim_event)

        return state

    # ------------------------------------------------------------------
    # Reflection phase
    # ------------------------------------------------------------------

    async def run_reflection(self, state: GameState, event_summary: str) -> GameState:
        """Run a reflection turn for all alive agents.

        Agents get to process what just happened (dawn results, vote
        results) and update their knowledge base.  No game actions are
        available -- only KB tools and pass_turn.

        Parameters
        ----------
        state:
            Current game state.
        event_summary:
            Text summary of what just happened.

        Returns
        -------
        GameState
            The (unchanged) state after reflection.
        """
        if not self.tool_factory or not self.briefing_builder:
            return state

        for player in state.get_alive_players():
            agent = self.agents.get(player.player_id)
            if agent is None:
                continue

            kb = self.knowledge_bases.get(player.player_id)
            if kb is None:
                continue

            self.tool_factory.state = state
            briefing = self.briefing_builder.build_reflection_briefing(
                state=state,
                player_id=player.player_id,
                kb=kb,
                event_summary=event_summary,
            )
            toolkit = self.tool_factory.build_reflection_toolkit(
                player_id=player.player_id,
            )

            try:
                await agent.run_phase(briefing, toolkit, max_rounds=3)
            except Exception:
                logger.exception(
                    "Agent %s raised during reflection", player.player_id
                )

        return state
