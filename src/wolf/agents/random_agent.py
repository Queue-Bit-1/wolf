"""Random-action baseline agent for benchmarking."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from wolf.agents.base import AgentBase
from wolf.engine.actions import (
    Action,
    NoAction,
    SpeakAction,
    UseAbilityAction,
    VoteAction,
)
from wolf.engine.events import GameEndEvent, GameEvent
from wolf.engine.phase import Phase

if TYPE_CHECKING:
    from wolf.agents.toolkit import AgentToolkit
    from wolf.engine.state import GameStateView

logger = logging.getLogger(__name__)

# Pre-built canned responses for the discussion phase.
_CANNED_RESPONSES = [
    "I think we should discuss more before voting.",
    "I'm not sure who to trust right now.",
    "Let's hear from everyone before jumping to conclusions.",
    "Something feels off, but I can't put my finger on it.",
    "I've been paying close attention and I have my suspicions.",
    "We need to focus on finding the werewolves.",
    "I'm a villager. Let's work together.",
    "Does anyone have useful information to share?",
    "I think we should be careful about who we vote for.",
    "Let's not rush this decision.",
]

_SUSPICION_TEMPLATES = [
    "I'm suspicious of {target}.",
    "I think {target} might be a werewolf.",
    "Has anyone else noticed {target} acting strange?",
    "{target} has been awfully quiet. That's suspicious.",
    "I don't trust {target}. Something about them feels off.",
]

_WOLF_RESPONSES = [
    "Let's target someone who seems dangerous.",
    "I think we should go after a quiet player.",
    "We need to eliminate the biggest threat.",
    "Any suggestions for tonight's target?",
]


class RandomAgent(AgentBase):
    """Agent that selects actions randomly.

    Supports both the new tool-based interface (``run_phase``) and the
    legacy interface (``decide_action``) for backward compatibility.
    """

    # ------------------------------------------------------------------
    # New tool-based interface
    # ------------------------------------------------------------------

    async def run_phase(self, briefing: str, toolkit: AgentToolkit, *, max_rounds: int | None = None) -> Action:
        """Pick a random terminal tool and invoke it with random args."""
        terminal_tools = toolkit.get_terminal_tools()
        if not terminal_tools:
            return NoAction(player_id=self.player_id, reason="no_terminal_tools")

        # Filter out pass_turn if there are other terminal tools
        action_tools = [t for t in terminal_tools if t.name != "pass_turn"]
        if not action_tools:
            action_tools = terminal_tools

        tool = random.choice(action_tools)
        args = self._generate_random_args(tool.name, briefing)

        result = toolkit.invoke(tool.name, args)
        if result.is_terminal and result.action is not None:
            return result.action

        # Fallback: try pass_turn
        fallback = toolkit.invoke("pass_turn", "")
        if fallback.is_terminal and fallback.action is not None:
            return fallback.action

        return NoAction(player_id=self.player_id, reason="random_fallback")

    def _generate_random_args(self, tool_name: str, briefing: str) -> str:
        """Generate random arguments for a tool based on its name."""
        if tool_name == "speak":
            return random.choice(_CANNED_RESPONSES)
        elif tool_name == "wolf_say":
            return random.choice(_WOLF_RESPONSES)
        elif tool_name in ("vote", "use_ability"):
            names = self._extract_names_from_briefing(briefing)
            if names:
                return random.choice(names)
            return "no_one"
        elif tool_name == "pass_turn":
            return ""
        return ""

    @staticmethod
    def _extract_names_from_briefing(briefing: str) -> list[str]:
        """Extract player names from briefing text.

        Tries several patterns: 'Valid vote targets:', 'Valid targets:',
        'Alive players:', to find a comma-separated name list.
        """
        import re
        for pattern in [
            r"Valid (?:vote )?targets?:\s*(.+)",
            r"Alive players?:\s*(.+)",
        ]:
            match = re.search(pattern, briefing)
            if match:
                names_str = match.group(1).strip()
                names = [n.strip() for n in names_str.split(",") if n.strip()]
                if names:
                    return names
        return []

    # ------------------------------------------------------------------
    # Legacy interface (backward compatibility with engine/game.py)
    # ------------------------------------------------------------------

    async def on_game_start(self, view: GameStateView) -> None:
        logger.info("RandomAgent %s joined the game", self.name)

    async def on_event(self, event: GameEvent) -> None:
        pass

    async def on_game_end(self, result: GameEndEvent) -> None:
        logger.info(
            "RandomAgent %s game ended. Winner: %s", self.name, result.winning_team
        )

    async def decide_action(self, view: GameStateView) -> Action:
        """Randomly select an action appropriate for the current phase."""
        if view.phase == Phase.DAY_DISCUSSION:
            return self._random_speak(view)
        elif view.phase == Phase.DAY_VOTE:
            return self._random_vote(view)
        elif view.phase == Phase.NIGHT:
            return self._random_night(view)
        else:
            return NoAction(player_id=self.player_id, reason="no_action_phase")

    def _random_speak(self, view: GameStateView) -> Action:
        alive_others = [
            p.name for p in view.alive_players if p.player_id != self.player_id
        ]
        if alive_others and random.random() < 0.5:
            target = random.choice(alive_others)
            content = random.choice(_SUSPICION_TEMPLATES).format(target=target)
        else:
            content = random.choice(_CANNED_RESPONSES)
        return SpeakAction(player_id=self.player_id, content=content)

    def _random_vote(self, view: GameStateView) -> Action:
        alive_others = [
            p for p in view.alive_players if p.player_id != self.player_id
        ]
        if not alive_others:
            return VoteAction(player_id=self.player_id, target_id=None)
        target = random.choice(alive_others)
        return VoteAction(player_id=self.player_id, target_id=target.player_id)

    def _random_night(self, view: GameStateView) -> Action:
        alive_others = [
            p for p in view.alive_players if p.player_id != self.player_id
        ]
        if not alive_others:
            return NoAction(player_id=self.player_id, reason="no_valid_targets")
        target = random.choice(alive_others)
        return UseAbilityAction(
            player_id=self.player_id,
            ability_name="",
            target_id=target.player_id,
        )
