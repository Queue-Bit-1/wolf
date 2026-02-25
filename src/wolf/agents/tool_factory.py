"""Builds per-agent, per-phase toolkits with information boundaries.

The ToolFactory holds a reference to the GameState (agents never do).
Tool handlers are closures that capture state + player context and
return only filtered text -- never structured objects.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from wolf.agents.toolkit import AgentToolkit, ToolDefinition
from wolf.engine.actions import (
    NoAction,
    SpeakAction,
    UseAbilityAction,
    VoteAction,
)

if TYPE_CHECKING:
    from wolf.agents.briefing_builder import BriefingBuilder
    from wolf.agents.knowledge_base import KnowledgeBase
    from wolf.engine.state import GameState

logger = logging.getLogger(__name__)


class ToolFactory:
    """Builds per-agent, per-phase toolkits.

    All tool handlers close over the factory's state references so
    agents never get direct access to ``GameState``.

    Parameters
    ----------
    state:
        Current game state (updated by moderator between phases).
    knowledge_bases:
        Mapping of player_id -> KnowledgeBase.
    id_to_name:
        Player ID to name mapping.
    name_to_id:
        Player name to ID mapping.
    briefing_builder:
        The briefing builder (used for vote history).
    """

    # Regex to strip <think>...</think> blocks from LLM output.
    _THINK_RE = __import__("re").compile(
        r"<think>.*?</think>", __import__("re").DOTALL
    )

    def __init__(
        self,
        state: GameState,
        knowledge_bases: dict[str, KnowledgeBase],
        id_to_name: dict[str, str],
        name_to_id: dict[str, str],
        briefing_builder: BriefingBuilder,
        randomize_names: bool = True,
    ) -> None:
        self.state = state
        self.knowledge_bases = knowledge_bases
        self._id_to_name = id_to_name
        self._name_to_id = name_to_id
        self._briefing_builder = briefing_builder
        self._randomize_names = randomize_names

    def _shuffle(self, names: list[str]) -> list[str]:
        """Return a shuffled copy of *names* when randomization is on."""
        if not self._randomize_names:
            return names
        shuffled = list(names)
        random.shuffle(shuffled)
        return shuffled

    def _name(self, player_id: str) -> str:
        return self._id_to_name.get(player_id, player_id)

    @classmethod
    def _clean_speech(cls, text: str) -> str:
        """Strip ``<think>...</think>`` blocks and excess whitespace.

        Thinking models (e.g. qwq) sometimes leak chain-of-thought
        tags into their public speech.  This ensures only the actual
        message reaches the game.
        """
        text = cls._THINK_RE.sub("", text).strip()
        return text

    @staticmethod
    def _extract_from_json(raw: str) -> str:
        """Try to extract a meaningful value from JSON-wrapped args.

        Models like gpt-oss sometimes wrap tool args in JSON, e.g.
        ``{"player_name":"Bob"}`` or ``{"name":"Alice","text":"..."}``.
        This extracts the first string value that looks like a player name.
        """
        import json

        raw = raw.strip()
        if not raw.startswith("{"):
            return raw
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                # Try common key names for player targets
                for key in ("player_name", "name", "target", "player"):
                    if key in data and isinstance(data[key], str):
                        return data[key]
                # Try common key names for text content
                for key in ("text", "message", "content", "note"):
                    if key in data and isinstance(data[key], str):
                        return data[key]
                # Fall back to first string value
                for v in data.values():
                    if isinstance(v, str):
                        return v
        except (json.JSONDecodeError, TypeError):
            pass
        return raw

    def _resolve_name(self, name_or_id: str) -> str | None:
        """Resolve a player name (case-insensitive) or ID to a player ID.

        Handles JSON-wrapped args from models like gpt-oss.
        """
        # First try to unwrap JSON
        name_or_id = self._extract_from_json(name_or_id)

        # Strip surrounding quotes that models sometimes add
        name_or_id = name_or_id.strip().strip("'\"").strip()

        # Direct ID match
        if name_or_id in self._id_to_name:
            return name_or_id
        # Name match (case-insensitive)
        name_lower = name_or_id.lower()
        for pname, pid in self._name_to_id.items():
            if pname.lower() == name_lower:
                return pid
        # Substring fallback
        for pname, pid in self._name_to_id.items():
            if name_lower in pname.lower() or pname.lower() in name_lower:
                return pid
        return None

    # ------------------------------------------------------------------
    # Always-available tools
    # ------------------------------------------------------------------

    def _build_common_tools(self, player_id: str) -> list[ToolDefinition]:
        """Tools available in every phase."""
        kb = self.knowledge_bases[player_id]
        state = self.state  # capture reference

        def get_alive_players(_args: str) -> str:
            alive = [self._name(p.player_id) for p in state.get_alive_players()]
            return "Alive players: " + ", ".join(self._shuffle(alive))

        def get_all_players(_args: str) -> str:
            entries = []
            for p in state.players:
                status = "alive" if p.is_alive else "eliminated"
                entries.append(f"  {self._name(p.player_id)}: {status}")
            return "All players:\n" + "\n".join(
                self._shuffle(entries) if self._randomize_names else entries
            )

        def get_day_events(_args: str) -> str:
            events = self._briefing_builder._get_public_day_events(state, player_id)
            if not events:
                return "(no events today)"
            return "Today's events:\n" + "\n".join(f"  {e}" for e in events)

        def get_vote_history(_args: str) -> str:
            return self._briefing_builder._get_vote_history(state)

        def read_notes(_args: str) -> str:
            return kb.read_notes()

        def write_notes(args: str) -> str:
            return kb.write_notes(ToolFactory._extract_from_json(args))

        def assess_player(args: str) -> str:
            return kb.assess_player(ToolFactory._extract_from_json(args))

        def read_assessments(_args: str) -> str:
            return kb.read_assessments()

        def pass_turn(_args: str):
            return NoAction(player_id=player_id, reason="pass_turn")

        return [
            ToolDefinition(
                name="get_alive_players",
                description="List all currently alive player names.",
                parameters="",
                is_terminal=False,
                handler=get_alive_players,
            ),
            ToolDefinition(
                name="get_all_players",
                description="List all players with their alive/eliminated status.",
                parameters="",
                is_terminal=False,
                handler=get_all_players,
            ),
            ToolDefinition(
                name="get_day_events",
                description="Get public events from the current day (night results, eliminations, etc.).",
                parameters="",
                is_terminal=False,
                handler=get_day_events,
            ),
            ToolDefinition(
                name="get_vote_history",
                description="View all past votes and results from previous days.",
                parameters="",
                is_terminal=False,
                handler=get_vote_history,
            ),
            ToolDefinition(
                name="read_notes",
                description="Read your personal scratchpad notes.",
                parameters="",
                is_terminal=False,
                handler=read_notes,
            ),
            ToolDefinition(
                name="write_notes",
                description="Append a note to your personal scratchpad.",
                parameters="text",
                is_terminal=False,
                handler=write_notes,
            ),
            ToolDefinition(
                name="assess_player",
                description="Record your assessment of a player. Format: 'player_name: assessment text'",
                parameters="name: text",
                is_terminal=False,
                handler=assess_player,
            ),
            ToolDefinition(
                name="read_assessments",
                description="Read all your player assessments.",
                parameters="",
                is_terminal=False,
                handler=read_assessments,
            ),
            ToolDefinition(
                name="pass_turn",
                description="Skip your turn / end your reflection.",
                parameters="",
                is_terminal=True,
                handler=pass_turn,
            ),
        ]

    # ------------------------------------------------------------------
    # Phase-specific toolkits
    # ------------------------------------------------------------------

    def build_discussion_toolkit(self, player_id: str) -> AgentToolkit:
        """Build toolkit for the discussion phase."""
        toolkit = AgentToolkit()
        for tool in self._build_common_tools(player_id):
            toolkit.register(tool)

        def speak(args: str):
            content = ToolFactory._clean_speech(
                ToolFactory._extract_from_json(args)
            )
            if not content:
                return NoAction(player_id=player_id, reason="empty_speech")
            return SpeakAction(player_id=player_id, content=content)

        toolkit.register(ToolDefinition(
            name="speak",
            description="Say something to the group during discussion.",
            parameters="message",
            is_terminal=True,
            handler=speak,
        ))
        return toolkit

    def build_vote_toolkit(self, player_id: str) -> AgentToolkit:
        """Build toolkit for the voting phase."""
        toolkit = AgentToolkit()
        for tool in self._build_common_tools(player_id):
            toolkit.register(tool)

        state = self.state

        def vote(args: str):
            target_raw = self._extract_from_json(args).strip()
            if not target_raw or target_raw.lower() in ("no_one", "no one", "none", "abstain"):
                return VoteAction(player_id=player_id, target_id=None)

            target_pid = self._resolve_name(target_raw)
            if target_pid is None:
                alive_names = self._shuffle([
                    self._name(p.player_id)
                    for p in state.get_alive_players()
                    if p.player_id != player_id
                ])
                return NoAction(
                    player_id=player_id,
                    reason=f"invalid_vote_target:{target_raw}. Valid: {', '.join(alive_names)}",
                )

            # Validate target is alive and not self
            player = state.get_player(target_pid)
            if player is None or not player.is_alive:
                return NoAction(player_id=player_id, reason=f"target_not_alive:{target_raw}")
            if target_pid == player_id:
                return NoAction(player_id=player_id, reason="cannot_vote_self")

            return VoteAction(player_id=player_id, target_id=target_pid)

        toolkit.register(ToolDefinition(
            name="vote",
            description="Vote to eliminate a player. Use 'no_one' to abstain.",
            parameters="player_name",
            is_terminal=True,
            handler=vote,
        ))
        return toolkit

    def build_night_toolkit(
        self,
        player_id: str,
        ability_name: str | None = None,
    ) -> AgentToolkit:
        """Build toolkit for the night phase (ability use).

        Parameters
        ----------
        player_id:
            The acting player.
        ability_name:
            The name of the player's night ability (if any).
        """
        toolkit = AgentToolkit()
        for tool in self._build_common_tools(player_id):
            toolkit.register(tool)

        state = self.state

        if ability_name:
            def use_ability(args: str):
                target_raw = self._extract_from_json(args).strip()
                if not target_raw:
                    return NoAction(player_id=player_id, reason="no_target_specified")

                target_pid = self._resolve_name(target_raw)
                if target_pid is None:
                    alive_names = self._shuffle([
                        self._name(p.player_id)
                        for p in state.get_alive_players()
                        if p.player_id != player_id
                    ])
                    return NoAction(
                        player_id=player_id,
                        reason=f"invalid_target:{target_raw}. Valid: {', '.join(alive_names)}",
                    )

                player = state.get_player(target_pid)
                if player is None or not player.is_alive:
                    return NoAction(player_id=player_id, reason=f"target_not_alive:{target_raw}")

                return UseAbilityAction(
                    player_id=player_id,
                    ability_name=ability_name,
                    target_id=target_pid,
                )

            toolkit.register(ToolDefinition(
                name="use_ability",
                description=f"Use your night ability ({ability_name}) on a player.",
                parameters="player_name",
                is_terminal=True,
                handler=use_ability,
            ))

        return toolkit

    def build_wolf_chat_toolkit(self, player_id: str) -> AgentToolkit:
        """Build toolkit for the wolf chat sub-phase.

        Raises ``ValueError`` if the player is not on the werewolf team
        (defence-in-depth — the moderator should already filter).
        """
        # Guard: only wolves get wolf_say
        player = self.state.get_player(player_id)
        if player is None or player.team != "werewolf":
            logger.error(
                "build_wolf_chat_toolkit called for non-wolf %s (team=%s)",
                player_id,
                player.team if player else "unknown",
            )
            raise ValueError(
                f"Player {player_id} is not a werewolf — cannot build wolf chat toolkit"
            )

        toolkit = AgentToolkit()
        for tool in self._build_common_tools(player_id):
            toolkit.register(tool)

        def wolf_say(args: str):
            content = ToolFactory._clean_speech(
                ToolFactory._extract_from_json(args)
            )
            if not content:
                return NoAction(player_id=player_id, reason="empty_wolf_message")
            return SpeakAction(player_id=player_id, content=content)

        toolkit.register(ToolDefinition(
            name="wolf_say",
            description="Send a message in the secret wolf chat channel.",
            parameters="message",
            is_terminal=True,
            handler=wolf_say,
        ))
        return toolkit

    def build_reflection_toolkit(self, player_id: str) -> AgentToolkit:
        """Build toolkit for the reflection phase (KB tools only, no actions).

        Uses the common tools which already include pass_turn as the
        only terminal tool.
        """
        toolkit = AgentToolkit()
        for tool in self._build_common_tools(player_id):
            toolkit.register(tool)
        return toolkit

    def build_game_start_toolkit(self, player_id: str) -> AgentToolkit:
        """Build toolkit for game start (KB tools + pass_turn only)."""
        toolkit = AgentToolkit()
        for tool in self._build_common_tools(player_id):
            toolkit.register(tool)
        return toolkit
