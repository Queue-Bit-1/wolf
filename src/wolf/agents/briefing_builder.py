"""Constructs curated text briefings for each agent per phase.

Each briefing only contains information the agent should legitimately
know.  No raw GameEvent objects are exposed -- only filtered text.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wolf.agents.knowledge_base import KnowledgeBase
    from wolf.engine.events import GameEvent
    from wolf.engine.state import GameState

logger = logging.getLogger(__name__)


class BriefingBuilder:
    """Builds curated text briefings for agents.

    The builder holds name/id mapping tables so it can translate
    player IDs to human-readable names in the briefing text.
    """

    def __init__(
        self,
        id_to_name: dict[str, str],
        name_to_id: dict[str, str],
        randomize_names: bool = True,
    ) -> None:
        self._id_to_name = id_to_name
        self._name_to_id = name_to_id
        self._randomize_names = randomize_names

    def _name(self, player_id: str) -> str:
        """Resolve a player_id to a display name."""
        return self._id_to_name.get(player_id, player_id)

    def _shuffle(self, names: list[str]) -> list[str]:
        """Return a shuffled copy of *names* when randomization is on."""
        if not self._randomize_names:
            return names
        shuffled = list(names)
        random.shuffle(shuffled)
        return shuffled

    # ------------------------------------------------------------------
    # Game start
    # ------------------------------------------------------------------

    def build_game_start_briefing(
        self,
        state: GameState,
        player_id: str,
        role_name: str,
        role_description: str,
        role_instructions: str,
        players: list[str],
    ) -> str:
        """Briefing delivered at game start.

        Parameters
        ----------
        state:
            Current game state.
        player_id:
            This agent's player ID.
        role_name:
            The agent's assigned role.
        role_description:
            Human-readable description of the role.
        role_instructions:
            Behavioural guidance for the role.
        players:
            List of all player names in the game.
        """
        name = self._name(player_id)
        player_list = "\n".join(f"  - {p}" for p in self._shuffle(players))

        return (
            f"=== Game Start ===\n"
            f"Welcome, {name}! A new game of Werewolf is beginning.\n\n"
            f"Your role: {role_name}\n"
            f"Role description: {role_description}\n"
            f"Role instructions: {role_instructions}\n\n"
            f"Players in the game:\n{player_list}\n\n"
            f"This is the game start phase -- use this time to take notes and "
            f"plan your strategy. Your role abilities (if any) will be available "
            f"during the night phase, not now.\n"
            f"Use write_notes and assess_player to prepare, then pass_turn when ready."
        )

    # ------------------------------------------------------------------
    # Night phase
    # ------------------------------------------------------------------

    def build_night_briefing(
        self,
        state: GameState,
        player_id: str,
        kb: KnowledgeBase,
        role_name: str,
        ability_name: str | None = None,
        ability_description: str | None = None,
        allies: list[str] | None = None,
        wolf_chat_messages: list[str] | None = None,
    ) -> str:
        """Briefing for the night phase (ability use).

        Parameters
        ----------
        state:
            Current game state.
        player_id:
            This agent's player ID.
        kb:
            The agent's knowledge base.
        role_name:
            The agent's role.
        ability_name:
            Name of the agent's night ability (if any).
        ability_description:
            Description of the ability.
        allies:
            Wolf allies (only for werewolf team).
        wolf_chat_messages:
            Messages from the wolf chat discussion (only for wolves).
        """
        name = self._name(player_id)
        alive = [self._name(p.player_id) for p in state.get_alive_players()]
        alive_others = [n for n in alive if n != name]

        lines = [
            f"=== Night Phase (Day {state.day}) ===",
            f"You are {name}, role: {role_name}.",
            "",
        ]

        if allies:
            lines.append(f"Your wolf allies: {', '.join(self._shuffle(allies))}")
            lines.append("")

        # Include wolf chat discussion so wolves remember what they agreed on
        if wolf_chat_messages:
            lines.append("=== Wolf Chat Discussion (just now) ===")
            for msg in wolf_chat_messages:
                lines.append(f"  {msg}")
            lines.append("")
            lines.append("Follow through on what your pack agreed to above.")
            lines.append("")

        if ability_name:
            lines.append(f"Your ability: {ability_name}")
            if ability_description:
                lines.append(f"  {ability_description}")
            lines.append(f"Valid targets: {', '.join(self._shuffle(alive_others))}")
            lines.append("")
            lines.append("Choose a target for your ability using the use_ability tool.")
        else:
            lines.append("You have no night ability. Use pass_turn to end your turn.")

        lines.append("")

        # KB summary
        kb_summary = kb.summarize_for_briefing()
        if kb_summary:
            lines.append(kb_summary)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Wolf chat
    # ------------------------------------------------------------------

    def build_wolf_chat_briefing(
        self,
        state: GameState,
        player_id: str,
        kb: KnowledgeBase,
        allies: list[str],
        prior_messages: list[str],
    ) -> str:
        """Briefing for the wolf chat sub-phase.

        Parameters
        ----------
        state:
            Current game state.
        player_id:
            This wolf's player ID.
        kb:
            The agent's knowledge base.
        allies:
            Names of wolf allies.
        prior_messages:
            Messages already posted in this wolf chat round.
        """
        name = self._name(player_id)
        alive = [self._name(p.player_id) for p in state.get_alive_players()]
        alive_villagers = [n for n in alive if n != name and n not in allies]

        lines = [
            f"=== Wolf Chat (Night {state.day}) ===",
            f"You are {name}. This is a private channel -- only wolves can see this.",
            f"Your wolf allies: {', '.join(self._shuffle(allies))}",
            "",
            f"Alive villager-side players: {', '.join(self._shuffle(alive_villagers)) if alive_villagers else '(none)'}",
            "",
        ]

        if prior_messages:
            lines.append("=== Wolf Chat Messages ===")
            for msg in prior_messages:
                lines.append(f"  {msg}")
            lines.append("")

        lines.append("Discuss strategy with your allies. Who should the pack target tonight?")
        lines.append("Use the wolf_say tool to speak, or pass_turn to stay silent.")
        lines.append("")

        kb_summary = kb.summarize_for_briefing()
        if kb_summary:
            lines.append(kb_summary)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Discussion phase
    # ------------------------------------------------------------------

    def build_discussion_briefing(
        self,
        state: GameState,
        player_id: str,
        kb: KnowledgeBase,
        role_name: str,
        speeches_so_far: list[tuple[str, str]],
    ) -> str:
        """Briefing for the discussion phase.

        Parameters
        ----------
        state:
            Current game state.
        player_id:
            This agent's player ID.
        kb:
            The agent's knowledge base.
        role_name:
            The agent's role.
        speeches_so_far:
            List of ``(player_name, message)`` tuples from earlier in this
            discussion round.
        """
        name = self._name(player_id)
        alive = [self._name(p.player_id) for p in state.get_alive_players()]
        all_players = [(self._name(p.player_id), p.is_alive) for p in state.players]

        lines = [
            f"=== Discussion Phase (Day {state.day}) ===",
            f"You are {name}, role: {role_name}.",
            "",
            "Alive players: " + ", ".join(self._shuffle(alive)),
            "",
        ]

        # Show dead players
        dead = [n for n, a in all_players if not a]
        if dead:
            lines.append("Eliminated players: " + ", ".join(self._shuffle(dead)))
            lines.append("")

        # Show day events (public events from earlier this day)
        day_events = self._get_public_day_events(state, player_id)
        if day_events:
            lines.append("=== Today's Events ===")
            for evt_text in day_events:
                lines.append(f"  {evt_text}")
            lines.append("")

        # Show speeches so far
        if speeches_so_far:
            lines.append("=== Discussion So Far ===")
            for speaker, message in speeches_so_far:
                lines.append(f"  [{speaker}]: {message}")
            lines.append("")

        lines.append("It's your turn to speak. Use the speak tool to address the group.")
        lines.append("You can also use tools to review information and update your notes.")
        lines.append("")

        kb_summary = kb.summarize_for_briefing()
        if kb_summary:
            lines.append(kb_summary)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Vote phase
    # ------------------------------------------------------------------

    def build_vote_briefing(
        self,
        state: GameState,
        player_id: str,
        kb: KnowledgeBase,
        role_name: str,
        valid_targets: list[str],
    ) -> str:
        """Briefing for the voting phase.

        Parameters
        ----------
        state:
            Current game state.
        player_id:
            This agent's player ID.
        kb:
            The agent's knowledge base.
        role_name:
            The agent's role.
        valid_targets:
            List of player names that can be voted for.
        """
        name = self._name(player_id)

        lines = [
            f"=== Voting Phase (Day {state.day}) ===",
            f"You are {name}, role: {role_name}.",
            "",
            f"Valid vote targets: {', '.join(self._shuffle(valid_targets))}",
            "(You may also vote for 'no_one' to abstain.)",
            "",
        ]

        # Show today's discussion speeches
        discussion_speeches = self._get_discussion_speeches(state)
        if discussion_speeches:
            lines.append("=== Discussion Summary ===")
            for speaker_id, content in discussion_speeches:
                speaker = self._name(speaker_id)
                lines.append(f"  [{speaker}]: {content}")
            lines.append("")

        # Show day events
        day_events = self._get_public_day_events(state, player_id)
        if day_events:
            lines.append("=== Today's Events ===")
            for evt_text in day_events:
                lines.append(f"  {evt_text}")
            lines.append("")

        lines.append("Cast your vote using the vote tool.")
        lines.append("")

        kb_summary = kb.summarize_for_briefing()
        if kb_summary:
            lines.append(kb_summary)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reflection phase (after dawn / vote results)
    # ------------------------------------------------------------------

    def build_reflection_briefing(
        self,
        state: GameState,
        player_id: str,
        kb: KnowledgeBase,
        event_summary: str,
    ) -> str:
        """Briefing for a reflection turn (after dawn or vote results).

        Agents get to process what happened and update their notes,
        but cannot take game actions.

        Parameters
        ----------
        state:
            Current game state.
        player_id:
            This agent's player ID.
        kb:
            The agent's knowledge base.
        event_summary:
            Text summary of what just happened (e.g. night results, vote outcome).
        """
        name = self._name(player_id)
        alive = [self._name(p.player_id) for p in state.get_alive_players()]

        lines = [
            f"=== Reflection ===",
            f"You are {name}.",
            "",
            f"Alive players: {', '.join(self._shuffle(alive))}",
            "",
            "=== What Just Happened ===",
            event_summary,
            "",
            "Take a moment to update your notes and assessments based on this information.",
            "Use write_notes and assess_player to record your thoughts.",
            "Use pass_turn when you're done reflecting.",
            "",
        ]

        kb_summary = kb.summarize_for_briefing()
        if kb_summary:
            lines.append(kb_summary)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_public_day_events(self, state: GameState, player_id: str) -> list[str]:
        """Extract public event text for the current day."""
        from wolf.engine.events import (
            EliminationEvent,
            NightResultEvent,
            PrivateRevealEvent,
            SpeechEvent,
            VoteEvent,
            VoteResultEvent,
        )

        result: list[str] = []
        for event in state.events:
            if event.day != state.day:
                continue

            if isinstance(event, NightResultEvent):
                if event.kills:
                    kill_names = [self._name(k) for k in event.kills]
                    text = f"Night results: {', '.join(kill_names)} killed."
                    if event.saved:
                        saved_names = [self._name(s) for s in event.saved]
                        text += f" {', '.join(saved_names)} saved."
                    result.append(text)

            elif isinstance(event, EliminationEvent):
                elim_name = self._name(event.player_id)
                # Standard Werewolf: eliminated players' roles are public
                result.append(
                    f"{elim_name} was eliminated (role: {event.role}, cause: {event.cause})."
                )

            elif isinstance(event, VoteResultEvent):
                if event.eliminated_id:
                    elim_name = self._name(event.eliminated_id)
                    result.append(f"Vote result: {elim_name} was voted out.")
                elif event.tie:
                    result.append("Vote result: tie -- no one was eliminated.")

            elif isinstance(event, PrivateRevealEvent):
                # Only the target sees their private info
                if event.player_id == player_id:
                    result.append(f"Private info: {event.info}")

            # Skip SpeechEvent and VoteEvent here -- they're shown in their
            # own sections of the briefing.

        return result

    def _get_discussion_speeches(self, state: GameState) -> list[tuple[str, str]]:
        """Extract discussion speeches from current day."""
        from wolf.engine.events import SpeechEvent
        from wolf.engine.phase import Phase

        speeches: list[tuple[str, str]] = []
        for event in state.events:
            if (
                isinstance(event, SpeechEvent)
                and event.day == state.day
                and event.phase == Phase.DAY_DISCUSSION
                and event.channel == "public"
            ):
                speeches.append((event.player_id, event.content))
        return speeches

    def _get_vote_history(self, state: GameState) -> str:
        """Build a text summary of all past votes."""
        from wolf.engine.events import VoteEvent, VoteResultEvent

        days: dict[int, list[str]] = {}
        for event in state.events:
            if isinstance(event, VoteEvent):
                voter = self._name(event.voter_id)
                target = self._name(event.target_id) if event.target_id else "no one"
                days.setdefault(event.day, []).append(f"  {voter} voted for {target}")
            elif isinstance(event, VoteResultEvent):
                if event.eliminated_id:
                    elim = self._name(event.eliminated_id)
                    days.setdefault(event.day, []).append(f"  Result: {elim} eliminated")
                elif event.tie:
                    days.setdefault(event.day, []).append("  Result: tie, no elimination")

        if not days:
            return "(no votes yet)"

        lines: list[str] = []
        for day in sorted(days):
            lines.append(f"Day {day}:")
            lines.extend(days[day])
        return "\n".join(lines)
