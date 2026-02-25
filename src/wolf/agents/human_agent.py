"""Human player agent that reads actions from stdin via tool commands."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from wolf.agents.base import AgentBase
from wolf.engine.actions import (
    Action,
    NoAction,
    SpeakAction,
    UseAbilityAction,
    VoteAction,
)
from wolf.engine.events import GameEndEvent, GameEvent, SpeechEvent, VoteEvent
from wolf.engine.phase import Phase

if TYPE_CHECKING:
    from wolf.agents.toolkit import AgentToolkit
    from wolf.engine.state import GameStateView

logger = logging.getLogger(__name__)


class HumanAgent(AgentBase):
    """Interactive agent that reads decisions from standard input.

    Supports both the new tool-based interface (``run_phase``) and the
    legacy interface (``decide_action``) for backward compatibility.
    """

    # ------------------------------------------------------------------
    # New tool-based interface
    # ------------------------------------------------------------------

    async def run_phase(self, briefing: str, toolkit: AgentToolkit, *, max_rounds: int | None = None) -> Action:
        """Print briefing + tools, then read USE: commands from stdin."""
        import re

        print(f"\n{'='*60}")
        print(briefing)
        print(f"\n{'='*60}")
        print(toolkit.format_for_prompt())
        print(f"{'='*60}")

        while True:
            line = await self._read_input("USE: ")
            line = line.strip()

            if not line:
                print("  (Enter a tool command, e.g. USE: speak(Hello everyone!))")
                continue

            text = line
            if text.upper().startswith("USE:"):
                text = text[4:].strip()

            match = re.match(r"(\w+)\((.*)\)$", text, re.DOTALL)
            if match:
                tool_name = match.group(1)
                args = match.group(2).strip()
            else:
                parts = text.split(None, 1)
                tool_name = parts[0] if parts else text
                args = parts[1] if len(parts) > 1 else ""

            result = toolkit.invoke(tool_name, args)
            print(f"  [{tool_name}] {result.output}")

            if result.is_terminal and result.action is not None:
                return result.action

    # ------------------------------------------------------------------
    # Legacy interface (backward compatibility with engine/game.py)
    # ------------------------------------------------------------------

    async def on_game_start(self, view: GameStateView) -> None:
        print(f"\n{'='*50}")
        print(f"  Welcome, {self.name}!")
        print(f"  Your role is: {view.my_player.role}")
        print(f"  Team: {view.my_player.team}")
        print(f"{'='*50}")
        print(f"\nPlayers in the game:")
        for pid, pname, alive in view.all_players:
            marker = " (you)" if pid == self.player_id else ""
            print(f"  - {pname}{marker}")
        print()

    async def on_event(self, event: GameEvent) -> None:
        if isinstance(event, SpeechEvent):
            print(f"  [{event.player_id}]: {event.content}")
        elif isinstance(event, VoteEvent):
            target = event.target_id if event.target_id else "no one"
            print(f"  [{event.voter_id}] voted for {target}")
        else:
            print(f"  [event] {event}")

    async def on_game_end(self, result: GameEndEvent) -> None:
        print(f"\n{'='*50}")
        print(f"  Game Over!")
        print(f"  Winning team: {result.winning_team}")
        print(f"  Winners: {', '.join(result.winners)}")
        print(f"  Reason: {result.reason}")
        print(f"{'='*50}\n")

    async def decide_action(self, view: GameStateView) -> Action:
        self._print_state_summary(view)
        if view.phase == Phase.DAY_DISCUSSION:
            return await self._prompt_speak(view)
        elif view.phase == Phase.DAY_VOTE:
            return await self._prompt_vote(view)
        elif view.phase == Phase.NIGHT:
            return await self._prompt_night(view)
        else:
            print("  (No action required this phase.)")
            return NoAction(player_id=self.player_id, reason="no_action_phase")

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------

    def _print_state_summary(self, view: GameStateView) -> None:
        print(f"\n--- Day {view.day} | {view.phase.name} ---")
        print(f"Alive players:")
        for p in view.alive_players:
            marker = " (you)" if p.player_id == self.player_id else ""
            print(f"  - {p.name}{marker}")
        print()

    async def _prompt_speak(self, view: GameStateView) -> Action:
        print("What would you like to say? (type your message)")
        line = await self._read_input("> ")
        if not line:
            return NoAction(player_id=self.player_id, reason="empty_input")
        return SpeakAction(player_id=self.player_id, content=line)

    async def _prompt_vote(self, view: GameStateView) -> Action:
        alive_others = [
            p.name for p in view.alive_players if p.player_id != self.player_id
        ]
        print("Who do you vote to eliminate?")
        for i, name in enumerate(alive_others, 1):
            print(f"  {i}. {name}")
        print(f"  0. No one (abstain)")

        line = await self._read_input("Vote> ")
        if not line or line.strip() == "0":
            return VoteAction(player_id=self.player_id, target_id=None)

        try:
            idx = int(line.strip()) - 1
            if 0 <= idx < len(alive_others):
                target = alive_others[idx]
                target_id = self._name_to_id_legacy(view, target)
                return VoteAction(player_id=self.player_id, target_id=target_id)
        except ValueError:
            pass

        target_id = self._name_to_id_legacy(view, line.strip())
        if target_id:
            return VoteAction(player_id=self.player_id, target_id=target_id)

        print(f"  Invalid target: {line.strip()}")
        return NoAction(player_id=self.player_id, reason="invalid_vote_input")

    async def _prompt_night(self, view: GameStateView) -> Action:
        alive_others = [
            p.name for p in view.alive_players if p.player_id != self.player_id
        ]
        print(f"Night phase. Your role: {view.my_player.role}")
        print("Choose a target for your ability:")
        for i, name in enumerate(alive_others, 1):
            print(f"  {i}. {name}")
        print(f"  0. Skip (no action)")

        line = await self._read_input("Target> ")
        if not line or line.strip() == "0":
            return NoAction(player_id=self.player_id, reason="skip_night")

        try:
            idx = int(line.strip()) - 1
            if 0 <= idx < len(alive_others):
                target = alive_others[idx]
                target_id = self._name_to_id_legacy(view, target)
                return UseAbilityAction(
                    player_id=self.player_id,
                    ability_name="",
                    target_id=target_id or "",
                )
        except ValueError:
            pass

        target_id = self._name_to_id_legacy(view, line.strip())
        if target_id:
            return UseAbilityAction(
                player_id=self.player_id,
                ability_name="",
                target_id=target_id,
            )

        print(f"  Invalid target: {line.strip()}")
        return NoAction(player_id=self.player_id, reason="invalid_night_input")

    @staticmethod
    def _name_to_id_legacy(view: GameStateView, name: str) -> str | None:
        name_lower = name.lower()
        for pid, pname, _alive in view.all_players:
            if pname.lower() == name_lower:
                return pid
        return None

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    @staticmethod
    async def _read_input(prompt: str) -> str:
        """Read a line from stdin without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: input(prompt))
