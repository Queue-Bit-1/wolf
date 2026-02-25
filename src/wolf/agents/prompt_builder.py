"""Prompt construction and response parsing for LLM agents."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from wolf.engine.actions import (
    Action,
    NoAction,
    SpeakAction,
    UseAbilityAction,
    VoteAction,
)
from wolf.engine.phase import Phase

if TYPE_CHECKING:
    from wolf.agents.memory import AgentMemory
    from wolf.engine.events import GameEvent, SpeechEvent
    from wolf.engine.state import GameStateView

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Constructs prompts for the two-call agent loop and parses responses."""

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        role_name: str,
        role_description: str,
        role_instructions: str,
        player_name: str,
    ) -> str:
        """Build the persistent system prompt that defines the agent's persona.

        Parameters
        ----------
        role_name:
            The role identifier (e.g. ``"werewolf"``, ``"seer"``).
        role_description:
            A human-readable description of the role.
        role_instructions:
            Role-specific behavioural guidance.
        player_name:
            The display name of the player.
        """
        return (
            "You are playing a game of Werewolf (also known as Mafia).\n"
            f"Your name is {player_name}.\n"
            f"Your role is: {role_name}.\n"
            f"\n"
            f"Role description:\n{role_description}\n"
            f"\n"
            f"Role-specific instructions:\n{role_instructions}\n"
            f"\n"
            "Behavioral guidelines:\n"
            "- Stay in character at all times.\n"
            "- Be strategic: think about what information you have and what "
            "others might know.\n"
            "- During discussion, try to be persuasive and gather information.\n"
            "- Pay attention to voting patterns and statements from other players.\n"
            "- If you are a villager-team role, try to identify the werewolves.\n"
            "- If you are a werewolf, try to blend in and avoid suspicion.\n"
            "- Keep your responses concise and relevant.\n"
            "- Never reveal your role unless it is strategically advantageous.\n"
        )

    # ------------------------------------------------------------------
    # Perception context
    # ------------------------------------------------------------------

    def build_perception_context(
        self,
        view: GameStateView,
        memory: AgentMemory,
        visible_messages: list[GameEvent],
    ) -> str:
        """Build a context block summarizing everything the agent can perceive.

        Parameters
        ----------
        view:
            The player's filtered game state view.
        memory:
            The agent's accumulated memory.
        visible_messages:
            Recent speech / event objects visible to this player.
        """
        lines: list[str] = []

        # Current state
        lines.append(f"=== Current Game State ===")
        lines.append(f"Day: {view.day}")
        lines.append(f"Phase: {view.phase.name}")
        lines.append(f"You are: {view.my_player.name} (role: {view.my_player.role})")
        lines.append(f"You are {'alive' if view.my_player.is_alive else 'dead'}.")
        lines.append("")

        # Alive players
        lines.append("=== Alive Players ===")
        for p in view.alive_players:
            marker = " (you)" if p.player_id == view.my_player.player_id else ""
            lines.append(f"- {p.name}{marker}")
        lines.append("")

        # All players status
        lines.append("=== All Players ===")
        for pid, pname, alive in view.all_players:
            status = "alive" if alive else "eliminated"
            marker = " (you)" if pid == view.my_player.player_id else ""
            lines.append(f"- {pname}: {status}{marker}")
        lines.append("")

        # Visible messages / events this phase
        if visible_messages:
            lines.append("=== Recent Messages ===")
            # Build ID->name lookup from alive players
            id_to_name: dict[str, str] = {}
            for p in view.alive_players:
                id_to_name[p.player_id] = p.name
            for pid, pname, _ in view.all_players:
                id_to_name[pid] = pname

            for evt in visible_messages:
                from wolf.engine.events import SpeechEvent, VoteEvent

                if isinstance(evt, SpeechEvent):
                    name = id_to_name.get(evt.player_id, evt.player_id)
                    lines.append(f"[{name}]: {evt.content}")
                elif isinstance(evt, VoteEvent):
                    voter = id_to_name.get(evt.voter_id, evt.voter_id)
                    target = id_to_name.get(evt.target_id, evt.target_id) if evt.target_id else "no one"
                    lines.append(f"[{voter}] voted for {target}")
                else:
                    lines.append(f"[event] {evt}")
            lines.append("")

        # Memory summary
        mem_summary = memory.summarize_for_prompt()
        if mem_summary and mem_summary != "(no memories yet)":
            lines.append(mem_summary)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reasoning prompt (call 1)
    # ------------------------------------------------------------------

    def build_reasoning_prompt(
        self,
        view: GameStateView,
        action_type: str,
    ) -> str:
        """Build a chain-of-thought prompt for strategic analysis.

        Parameters
        ----------
        view:
            The player's filtered game state view.
        action_type:
            One of ``"night_ability"``, ``"discussion"``, ``"vote"``.
        """
        if action_type == "night_ability":
            return (
                "It is night. You must decide how to use your ability.\n"
                "Think step by step:\n"
                "1. What information do you currently have about each player?\n"
                "2. Who is most suspicious or most valuable to target?\n"
                "3. What would be the most strategic use of your ability tonight?\n"
                "4. Consider what other players might do tonight.\n"
                "\n"
                "Provide your strategic reasoning."
            )

        if action_type == "discussion":
            return (
                "It is the discussion phase. You will speak to the group.\n"
                "Think step by step:\n"
                "1. What do you know so far about other players?\n"
                "2. What happened last night or in previous rounds?\n"
                "3. Who seems suspicious and why?\n"
                "4. What information should you share or hide?\n"
                "5. What is your strategy for this discussion?\n"
                "\n"
                "Provide your strategic reasoning."
            )

        if action_type == "vote":
            return (
                "It is time to vote on who to eliminate.\n"
                "Think step by step:\n"
                "1. Review what was said during the discussion phase.\n"
                "2. Who is most suspicious based on behavior and statements?\n"
                "3. What are the voting dynamics -- who might others vote for?\n"
                "4. Is it better to vote with the majority or go against it?\n"
                "5. Who should you vote for and why?\n"
                "\n"
                "Provide your strategic reasoning."
            )

        return "Analyze the current game situation and decide on your next move."

    # ------------------------------------------------------------------
    # Action prompt (call 2)
    # ------------------------------------------------------------------

    def build_action_prompt(
        self,
        view: GameStateView,
        action_type: str,
        valid_targets: list[str],
    ) -> str:
        """Build a structured-output prompt requesting a concrete action.

        Parameters
        ----------
        view:
            The player's filtered game state view.
        action_type:
            One of ``"night_ability"``, ``"discussion"``, ``"vote"``.
        valid_targets:
            List of valid target identifiers for the action.
        """
        if action_type == "discussion":
            return (
                "Based on your reasoning, compose your message to the group.\n"
                "Keep it concise (a few sentences).\n"
                "\n"
                "Respond with EXACTLY this format:\n"
                "SPEAK: <your message>\n"
            )

        if action_type == "vote":
            target_list = ", ".join(valid_targets) if valid_targets else "(none)"
            return (
                "Based on your reasoning, cast your vote.\n"
                f"Valid targets: {target_list}\n"
                "You may also vote for no one.\n"
                "\n"
                "Respond with EXACTLY one of:\n"
                "VOTE: <player_name>\n"
                "VOTE: no_one\n"
            )

        if action_type == "night_ability":
            target_list = ", ".join(valid_targets) if valid_targets else "(none)"
            return (
                "Based on your reasoning, choose a target for your ability.\n"
                f"Valid targets: {target_list}\n"
                "\n"
                "Respond with EXACTLY this format:\n"
                "TARGET: <player_name>\n"
            )

        return "Decide on your action."

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_action_response(
        self,
        response: str,
        action_type: str,
        player_id: str,
        valid_targets: list[str],
    ) -> Action:
        """Parse a raw LLM response into a typed Action.

        Falls back to :class:`NoAction` for malformed or unrecognisable
        responses.

        Parameters
        ----------
        response:
            Raw text from the LLM completion.
        action_type:
            One of ``"night_ability"``, ``"discussion"``, ``"vote"``.
        player_id:
            The acting player's id.
        valid_targets:
            Acceptable target names/ids for validation.
        """
        text = response.strip()

        if action_type == "discussion":
            return self._parse_speak(text, player_id)

        if action_type == "vote":
            return self._parse_vote(text, player_id, valid_targets)

        if action_type == "night_ability":
            return self._parse_ability(text, player_id, valid_targets)

        logger.warning("Unknown action_type %r, returning NoAction", action_type)
        return NoAction(player_id=player_id, reason="unknown_action_type")

    # ------ private parsers ------

    def _parse_speak(self, text: str, player_id: str) -> Action:
        match = re.search(r"SPEAK:\s*(.+)", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return SpeakAction(player_id=player_id, content=content)

        # Fallback: treat entire response as speech if it looks non-empty.
        if text and len(text) > 3:
            logger.info("No SPEAK: prefix found, using raw response as speech")
            return SpeakAction(player_id=player_id, content=text)

        logger.warning("Could not parse speak response, returning NoAction")
        return NoAction(player_id=player_id, reason="unparseable_speak")

    def _parse_vote(self, text: str, player_id: str, valid_targets: list[str]) -> Action:
        match = re.search(r"VOTE:\s*(.+)", text)
        if match:
            target_raw = match.group(1).strip().lower()

            if target_raw in ("no_one", "no one", "none", "abstain"):
                return VoteAction(player_id=player_id, target_id=None)

            # Try exact or case-insensitive match against valid targets.
            target_id = self._fuzzy_match_target(target_raw, valid_targets)
            if target_id is not None:
                return VoteAction(player_id=player_id, target_id=target_id)

            logger.warning(
                "Vote target %r not in valid targets %s, returning NoAction",
                target_raw,
                valid_targets,
            )
            return NoAction(player_id=player_id, reason=f"invalid_vote_target:{target_raw}")

        logger.warning("Could not parse vote response, returning NoAction")
        return NoAction(player_id=player_id, reason="unparseable_vote")

    def _parse_ability(self, text: str, player_id: str, valid_targets: list[str]) -> Action:
        match = re.search(r"TARGET:\s*(.+)", text)
        if match:
            target_raw = match.group(1).strip().lower()
            target_id = self._fuzzy_match_target(target_raw, valid_targets)
            if target_id is not None:
                return UseAbilityAction(
                    player_id=player_id,
                    ability_name="",  # filled by caller
                    target_id=target_id,
                )

            logger.warning(
                "Ability target %r not in valid targets %s, returning NoAction",
                target_raw,
                valid_targets,
            )
            return NoAction(player_id=player_id, reason=f"invalid_ability_target:{target_raw}")

        logger.warning("Could not parse ability response, returning NoAction")
        return NoAction(player_id=player_id, reason="unparseable_ability")

    @staticmethod
    def _fuzzy_match_target(raw: str, valid_targets: list[str]) -> str | None:
        """Try to match *raw* against *valid_targets* (case-insensitive)."""
        raw_lower = raw.lower().strip()
        for target in valid_targets:
            if target.lower() == raw_lower:
                return target
        # Substring match as last resort.
        for target in valid_targets:
            if raw_lower in target.lower() or target.lower() in raw_lower:
                return target
        return None
