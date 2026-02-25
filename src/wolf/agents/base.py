"""Abstract base class for all game agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from wolf.engine.actions import Action

if TYPE_CHECKING:
    from wolf.agents.toolkit import AgentToolkit
    from wolf.engine.events import GameEndEvent, GameEvent


class AgentBase(ABC):
    """Interface that every agent (LLM, human, random) must implement.

    The new tool-based architecture uses a single entry point:

    * :meth:`run_phase` -- called each time the agent must act, with a
      text briefing and a toolkit of callable tools.

    Legacy lifecycle methods are kept for backward compatibility but are
    no longer called by the moderator:

    * :meth:`on_game_start`
    * :meth:`decide_action`
    * :meth:`on_event`
    * :meth:`on_game_end`
    """

    def __init__(self, player_id: str, name: str) -> None:
        self.player_id = player_id
        self.name = name

    # ------------------------------------------------------------------
    # New tool-based interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run_phase(
        self,
        briefing: str,
        toolkit: AgentToolkit,
        *,
        max_rounds: int | None = None,
    ) -> Action:
        """Execute the agent's turn for one phase.

        Parameters
        ----------
        briefing:
            Curated text describing the current game state and what
            the agent should do.  Contains only information the agent
            is allowed to know.
        toolkit:
            A set of callable tools the agent can invoke during this
            turn.  Terminal tools (e.g. speak, vote, use_ability)
            end the turn and return an Action.
        max_rounds:
            Optional cap on the number of ReAct rounds.  If ``None``,
            the agent uses its own default.

        Returns
        -------
        Action
            The terminal action produced by invoking a terminal tool.
        """
        ...

    # ------------------------------------------------------------------
    # Legacy interface (kept for backward compatibility)
    # ------------------------------------------------------------------

    async def on_game_start(self, view) -> None:
        """Called when the game starts with the initial state view."""
        pass

    async def decide_action(self, view) -> Action:
        """Decide what action to take given the current game state view."""
        from wolf.engine.actions import NoAction
        return NoAction(player_id=self.player_id, reason="legacy_not_implemented")

    async def on_event(self, event: GameEvent) -> None:
        """Called when a game event occurs that this player can see."""
        pass

    async def on_game_end(self, result: GameEndEvent) -> None:
        """Called when the game ends."""
        pass
