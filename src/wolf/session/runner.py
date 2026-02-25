"""Single-game runner that orchestrates setup, execution, and metrics."""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from wolf.config.schema import GameConfig, ModelConfig, PlayerConfig
from wolf.engine.events import GameEndEvent
from wolf.engine.state import GameState, PlayerSlot
from wolf.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single game execution."""

    game_id: str
    config: GameConfig
    end_event: GameEndEvent
    game_summary: dict[str, Any]
    duration: float
    model_map: dict[str, str] = field(default_factory=dict)


class GameRunner:
    """Runs a single Werewolf game from config to completion.

    Creates the role registry, assigns roles to players, instantiates
    agents, sets up the tool-based architecture, and drives the game.
    """

    def __init__(
        self,
        config: GameConfig,
        extra_listeners: list[Any] | None = None,
        game_number: int = 0,
        cross_game_memories: dict[str, list[str]] | None = None,
    ) -> None:
        self.config = config
        self.extra_listeners = extra_listeners or []
        self.game_number = game_number
        self.cross_game_memories = cross_game_memories or {}

    async def run(self) -> GameResult:
        """Execute a single game and return the result."""
        game_id = str(uuid.uuid4())
        start_time = time.monotonic()

        logger.info("Starting game %s (%s)", game_id, self.config.game_name)

        # 1. Discover roles
        from wolf.roles.registry import RoleRegistry

        RoleRegistry.discover_plugins()

        # 2. Build player slots with role assignment
        player_slots, role_assignments = _assign_roles(self.config)

        # 3. Build initial game state
        state = GameState(
            day=0,
            players=tuple(player_slots),
        )

        # 4. Create agents (with model_pool support)
        agents, model_map = _create_agents(
            self.config, player_slots, role_assignments
        )

        # 4b. Validate context length for all LLM agents
        await _validate_context_lengths(agents)

        # 5. Create channel manager
        wolf_ids = [
            ps.player_id for ps in player_slots if ps.team == "werewolf"
        ]
        all_ids = [ps.player_id for ps in player_slots]
        channel_manager = _create_channel_manager(
            all_ids, wolf_ids, self.config
        )

        # 6. Create metrics collector
        collector = MetricsCollector()
        for ps in player_slots:
            collector.register_player(
                player_id=ps.player_id,
                name=ps.name,
                role=ps.role,
                team=ps.team,
            )

        # 7. Build name/id mappings
        id_to_name = {ps.player_id: ps.name for ps in player_slots}
        name_to_id: dict[str, str] = {}
        for ps in player_slots:
            name_to_id[ps.name] = ps.player_id
            name_to_id[ps.name.lower()] = ps.player_id

        # 8. Create KnowledgeBases
        from wolf.agents.knowledge_base import KnowledgeBase

        knowledge_bases: dict[str, KnowledgeBase] = {}
        for ps in player_slots:
            kb = KnowledgeBase(player_name=ps.name)
            # Inject cross-game learnings
            prior = self.cross_game_memories.get(ps.name, [])
            if prior:
                kb.inject_learnings(prior)
            knowledge_bases[ps.player_id] = kb

        # 9. Create BriefingBuilder and ToolFactory
        from wolf.agents.briefing_builder import BriefingBuilder
        from wolf.agents.tool_factory import ToolFactory

        briefing_builder = BriefingBuilder(
            id_to_name=id_to_name,
            name_to_id=name_to_id,
            randomize_names=self.config.randomize_names,
        )
        tool_factory = ToolFactory(
            state=state,
            knowledge_bases=knowledge_bases,
            id_to_name=id_to_name,
            name_to_id=name_to_id,
            briefing_builder=briefing_builder,
            randomize_names=self.config.randomize_names,
        )

        # 10. Create and run the game
        from wolf.engine.moderator import Moderator
        from wolf.engine.phase import Phase
        from wolf.engine.victory import check_victory
        from wolf.engine.events import GameEndEvent as _GEE, PhaseChangeEvent
        from wolf.narrator import Narrator, set_player_info

        # Set up live narrator
        narrator = Narrator()
        set_player_info(
            names=id_to_name,
            models=model_map,
            roles={ps.player_id: ps.role for ps in player_slots},
        )

        event_listeners: list[Any] = [collector, narrator] + self.extra_listeners

        # Notify extra listeners about game info
        roles_map = {ps.player_id: ps.role for ps in player_slots}
        teams_map = {ps.player_id: ps.team for ps in player_slots}
        for listener in self.extra_listeners:
            if hasattr(listener, "set_game_info"):
                listener.set_game_info(
                    game_number=self.game_number,
                    names=id_to_name,
                    models=model_map,
                    roles=roles_map,
                    teams=teams_map,
                )

        # Print setup table
        game_label = (
            f"GAME {self.game_number}: {self.config.game_name}"
            if self.game_number
            else f"WOLF BENCHMARK: {self.config.game_name}"
        )
        print(f"\n\033[1m{'='*55}\033[0m")
        print(f"\033[1m  {game_label}\033[0m")
        print(f"  {'─'*51}")
        print(f"  {'Player':<10} {'Model':<20} {'Role':<12} {'Team'}")
        print(f"  {'─'*51}")
        for ps in player_slots:
            model_name = model_map.get(ps.player_id, self.config.default_model.model)
            team_color = "\033[91m" if ps.team == "werewolf" else "\033[92m"
            print(
                f"  {ps.name:<10} {model_name:<20} "
                f"{team_color}{ps.role:<12}{ps.team}\033[0m"
            )
        print(f"\033[1m{'='*55}\033[0m", flush=True)

        moderator = Moderator(
            state=state,
            agents=agents,
            channel_manager=channel_manager,
            event_listeners=event_listeners,
            config=self.config,
            role_registry=RoleRegistry,
            tool_factory=tool_factory,
            briefing_builder=briefing_builder,
            knowledge_bases=knowledge_bases,
        )

        # Wire up event callback on LLM agents so ReasoningEvents get emitted
        for agent in agents.values():
            if hasattr(agent, "_emit_event"):
                agent._emit_event = moderator.emit_event

        # Run game start briefings for all agents
        for ps in player_slots:
            agent = agents.get(ps.player_id)
            if agent is None:
                continue

            role = RoleRegistry.get(role_assignments[ps.player_id])
            player_names = [p.name for p in player_slots]

            tool_factory.state = state
            briefing = briefing_builder.build_game_start_briefing(
                state=state,
                player_id=ps.player_id,
                role_name=role.name,
                role_description=role.description,
                role_instructions=role.prompt_instructions,
                players=player_names,
            )
            toolkit = tool_factory.build_game_start_toolkit(ps.player_id)

            try:
                await agent.run_phase(briefing, toolkit, max_rounds=3)
            except Exception:
                logger.exception(
                    "Agent %s raised during game start", ps.player_id
                )

        # Run the game loop
        end_event = await _run_game_loop(moderator, self.config)

        duration = time.monotonic() - start_time

        # Collect summary
        game_summary = collector.get_game_summary()

        # Attach model info to player summaries
        for ps_summary in game_summary.get("players", []):
            pid = ps_summary.get("player_id", "")
            ps_summary["model"] = model_map.get(pid, self.config.default_model.model)

        # Print token usage summary
        print(f"\n\033[1m  TOKEN USAGE\033[0m")
        print(f"  {'─'*51}")
        print(f"  {'Player':<10} {'Model':<20} {'In':>8} {'Out':>8}")
        print(f"  {'─'*51}")
        total_in = total_out = 0
        for ps in player_slots:
            agent = agents.get(ps.player_id)
            tracker = getattr(agent, "token_tracker", None)
            if tracker:
                usage = tracker.get_player_usage(ps.player_id)
                t_in = usage["total_input"]
                t_out = usage["total_output"]
                total_in += t_in
                total_out += t_out
                mn = model_map.get(ps.player_id, self.config.default_model.model)
                print(f"  {ps.name:<10} {mn:<20} {t_in:>8} {t_out:>8}")
        print(f"  {'─'*51}")
        print(f"  {'TOTAL':<10} {'':<20} {total_in:>8} {total_out:>8}")
        print(f"  Duration: {duration:.1f}s", flush=True)

        # Extract cross-game learnings from KnowledgeBases
        for ps in player_slots:
            kb = knowledge_bases.get(ps.player_id)
            if kb:
                learnings = kb.extract_learnings(self.game_number)
                existing = self.cross_game_memories.setdefault(ps.name, [])
                existing.extend(learnings)

        logger.info(
            "Game %s completed in %.1fs - %s wins",
            game_id,
            duration,
            end_event.winning_team,
        )

        return GameResult(
            game_id=game_id,
            config=self.config,
            end_event=end_event,
            game_summary=game_summary,
            duration=duration,
            model_map=model_map,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _assign_roles(
    config: GameConfig,
) -> tuple[list[PlayerSlot], dict[str, str]]:
    """Create player slots with randomly assigned roles."""
    from wolf.roles.registry import RoleRegistry

    # Expand role slots into a list of role names
    role_list: list[str] = []
    for slot in config.roles:
        role_list.extend([slot.role] * slot.count)

    # Pad or trim to match num_players
    while len(role_list) < config.num_players:
        role_list.append("villager")
    role_list = role_list[: config.num_players]

    # Shuffle roles
    random.shuffle(role_list)

    # Generate or use configured player names
    player_configs = list(config.players)
    while len(player_configs) < config.num_players:
        idx = len(player_configs) + 1
        player_configs.append(
            PlayerConfig(name=f"Player_{idx}", agent_type="llm")
        )

    player_slots: list[PlayerSlot] = []
    role_assignments: dict[str, str] = {}

    for i, (role_name, pc) in enumerate(zip(role_list, player_configs)):
        player_id = f"p{i + 1}"
        role_instance = RoleRegistry.get(role_name)

        slot = PlayerSlot(
            player_id=player_id,
            name=pc.name,
            role=role_name,
            team=role_instance.team,
        )
        player_slots.append(slot)
        role_assignments[player_id] = role_name

    return player_slots, role_assignments


async def _validate_context_lengths(agents: dict[str, Any]) -> None:
    """Validate that context length settings are respected by Ollama.

    Sends a test request per unique model to confirm ``num_ctx`` is
    being applied, catching silent fallbacks to the 2048 default.
    """
    from wolf.agents.llm_agent import LLMAgent

    validated: set[str] = set()
    for agent in agents.values():
        if not isinstance(agent, LLMAgent):
            continue
        model_key = f"{agent.model_config.model}@{agent.model_config.context_length}"
        if model_key in validated:
            continue
        validated.add(model_key)

        ctx = await agent.client.validate_context_length()
        configured = agent.model_config.context_length
        print(
            f"  \033[2mContext length: {agent.model_config.model} "
            f"-> num_ctx={configured} ✓\033[0m",
            flush=True,
        )


def _create_agents(
    config: GameConfig,
    player_slots: list[PlayerSlot],
    role_assignments: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Instantiate agent objects for each player based on config."""
    agents: dict[str, Any] = {}
    model_map: dict[str, str] = {}

    player_configs = list(config.players)
    while len(player_configs) < len(player_slots):
        idx = len(player_configs) + 1
        player_configs.append(
            PlayerConfig(name=f"Player_{idx}", agent_type="llm")
        )

    for ps, pc in zip(player_slots, player_configs):
        agent_type = pc.agent_type.lower()

        # Determine model: per-player override > model_pool random > default
        if pc.model:
            model_config = pc.model
        elif config.model_pool:
            model_config = random.choice(config.model_pool)
        else:
            model_config = config.default_model

        model_map[ps.player_id] = model_config.model

        try:
            if agent_type == "human":
                from wolf.agents.human_agent import HumanAgent

                agents[ps.player_id] = HumanAgent(
                    player_id=ps.player_id,
                    name=ps.name,
                )
            elif agent_type == "random":
                from wolf.agents.random_agent import RandomAgent

                agents[ps.player_id] = RandomAgent(
                    player_id=ps.player_id,
                    name=ps.name,
                )
            else:
                # Default to LLM agent
                from wolf.agents.llm_agent import LLMAgent
                from wolf.roles.registry import RoleRegistry as _RR

                role_obj = _RR.get(role_assignments[ps.player_id])
                agents[ps.player_id] = LLMAgent(
                    player_id=ps.player_id,
                    name=ps.name,
                    model_config=model_config,
                    role=role_obj,
                )
        except ImportError:
            logger.warning(
                "Agent type %r not available, using stub for %s",
                agent_type,
                ps.player_id,
            )
            agents[ps.player_id] = _StubAgent(
                player_id=ps.player_id, name=ps.name
            )

    return agents, model_map


def _create_channel_manager(
    all_ids: list[str],
    wolf_ids: list[str],
    config: GameConfig,
) -> Any:
    """Create a ChannelManager. Falls back to a stub if unavailable."""
    try:
        from wolf.comms.manager import ChannelManager

        manager = ChannelManager()
        manager.create_channels(all_ids, wolf_ids, config.communication)
        return manager
    except (ImportError, AttributeError):
        logger.warning("ChannelManager not available, using stub")
        return _StubChannelManager()


def _build_event_summary(
    state: GameState,
    events: list,
    id_to_name: dict[str, str],
) -> str:
    """Build a text summary of recent events for reflection briefings."""
    from wolf.engine.events import (
        EliminationEvent,
        NightResultEvent,
        VoteResultEvent,
    )

    parts: list[str] = []
    for event in events:
        if isinstance(event, NightResultEvent):
            if event.kills:
                kill_names = [id_to_name.get(k, k) for k in event.kills]
                text = f"During the night, {', '.join(kill_names)} were killed."
                if event.saved:
                    saved_names = [id_to_name.get(s, s) for s in event.saved]
                    text += f" {', '.join(saved_names)} were saved by the doctor."
                parts.append(text)
        elif isinstance(event, EliminationEvent):
            name = id_to_name.get(event.player_id, event.player_id)
            parts.append(
                f"{name} was eliminated (role: {event.role}, cause: {event.cause})."
            )
        elif isinstance(event, VoteResultEvent):
            if event.eliminated_id:
                name = id_to_name.get(event.eliminated_id, event.eliminated_id)
                parts.append(f"The vote resulted in {name} being eliminated.")
            elif event.tie:
                parts.append("The vote was a tie -- no one was eliminated.")
            else:
                parts.append("No one received enough votes to be eliminated.")

    return "\n".join(parts) if parts else "Nothing notable happened."


async def _run_game_loop(
    moderator: Any,
    config: GameConfig,
) -> GameEndEvent:
    """Drive the game through phases until a victory condition is met."""
    from wolf.engine.phase import Phase
    from wolf.engine.victory import check_victory
    from wolf.engine.events import PhaseChangeEvent

    state = moderator.state
    id_to_name = {}
    if moderator.tool_factory:
        id_to_name = moderator.tool_factory._id_to_name

    for day in range(1, config.max_days + 1):
        state = state.with_day(day)

        # --- NIGHT ---
        state = state.with_phase(Phase.NIGHT)
        moderator.state = state
        moderator.emit_event(
            PhaseChangeEvent(
                day=day, phase=Phase.NIGHT,
                old_phase=Phase.SETUP if day == 1 else Phase.DAY_VOTE_RESULT,
                new_phase=Phase.NIGHT,
            )
        )
        state = await moderator.run_night(state)

        # --- DAWN (resolve night) ---
        state = state.with_phase(Phase.DAWN)
        moderator.state = state
        moderator.emit_event(
            PhaseChangeEvent(
                day=day, phase=Phase.DAWN,
                old_phase=Phase.NIGHT, new_phase=Phase.DAWN,
            )
        )

        # Capture events before dawn to detect new events
        pre_dawn_count = len(state.events)
        state = await moderator.run_dawn(state)

        # Check victory after night kills
        end = check_victory(state)
        if end is not None:
            moderator.emit_event(end)
            return end

        # --- Dawn Reflection ---
        dawn_events = list(state.events[pre_dawn_count:])
        if dawn_events:
            dawn_summary = _build_event_summary(state, dawn_events, id_to_name)
            moderator.state = state
            state = await moderator.run_reflection(state, dawn_summary)

        # --- DAY DISCUSSION ---
        state = state.with_phase(Phase.DAY_DISCUSSION)
        moderator.state = state
        moderator.emit_event(
            PhaseChangeEvent(
                day=day, phase=Phase.DAY_DISCUSSION,
                old_phase=Phase.DAWN, new_phase=Phase.DAY_DISCUSSION,
            )
        )
        state = await moderator.run_discussion(state)

        # --- DAY VOTE ---
        state = state.with_phase(Phase.DAY_VOTE)
        moderator.state = state
        moderator.emit_event(
            PhaseChangeEvent(
                day=day, phase=Phase.DAY_VOTE,
                old_phase=Phase.DAY_DISCUSSION, new_phase=Phase.DAY_VOTE,
            )
        )
        pre_vote_count = len(state.events)
        state = await moderator.run_vote(state)

        # --- VOTE RESULT ---
        from wolf.engine.events import VoteResultEvent

        vote_result_event = None
        for ev in reversed(state.events):
            if isinstance(ev, VoteResultEvent):
                vote_result_event = ev
                break

        if vote_result_event is not None:
            state = state.with_phase(Phase.DAY_VOTE_RESULT)
            moderator.state = state
            state = await moderator.run_vote_result(state, vote_result_event)

        # Check victory after vote elimination
        end = check_victory(state)
        if end is not None:
            moderator.emit_event(end)
            return end

        # --- Vote Reflection ---
        vote_events = list(state.events[pre_vote_count:])
        if vote_events:
            vote_summary = _build_event_summary(state, vote_events, id_to_name)
            moderator.state = state
            state = await moderator.run_reflection(state, vote_summary)

    # Game hit max_days without resolution
    end = GameEndEvent(
        day=config.max_days,
        winning_team="draw",
        winners=[],
        reason=f"Game reached maximum of {config.max_days} days.",
    )
    moderator.emit_event(end)
    return end


# ------------------------------------------------------------------
# Stubs for when dependencies are not yet available
# ------------------------------------------------------------------


class _StubAgent:
    """Minimal agent stub that always takes no action."""

    def __init__(self, player_id: str, name: str) -> None:
        self.player_id = player_id
        self.name = name

    async def run_phase(self, briefing: str, toolkit: Any) -> Any:
        from wolf.engine.actions import NoAction
        return NoAction(player_id=self.player_id, reason="stub_agent")

    async def on_game_start(self, view: Any) -> None:
        pass

    async def decide_action(self, game_state_view: Any) -> Any:
        from wolf.engine.actions import NoAction
        return NoAction(player_id=self.player_id, reason="stub_agent")

    async def on_event(self, event: Any) -> None:
        pass

    async def on_game_end(self, result: Any) -> None:
        pass


class _StubChannelManager:
    """Minimal channel manager stub."""

    def create_channels(self, *args: Any, **kwargs: Any) -> None:
        pass
