"""Agent memory system: observations, player models, and decision history."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PlayerModel:
    """Mental model of another player maintained by an agent."""

    player_id: str
    name: str
    suspicion: float = 0.5
    trust: float = 0.5
    notes: list[str] = field(default_factory=list)
    voted_for: list[str] = field(default_factory=list)
    voted_by: list[str] = field(default_factory=list)
    claimed_role: str | None = None


@dataclass
class Observation:
    """A single observed fact or event."""

    day: int
    phase: str
    content: str
    importance: float = 0.5
    source: str = ""


class AgentMemory:
    """Working memory for an LLM agent.

    Stores observations from the game, mental models of other players,
    and a log of the agent's own decisions.
    """

    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.player_models: dict[str, PlayerModel] = {}
        self.decisions: list[dict] = []

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def add_observation(self, obs: Observation) -> None:
        """Record a new observation."""
        self.observations.append(obs)
        logger.debug("Observation added: %s (importance=%.2f)", obs.content[:60], obs.importance)

    def get_recent_observations(self, n: int = 10) -> list[Observation]:
        """Return the *n* most recent observations (newest last)."""
        return self.observations[-n:]

    def get_important_observations(self, threshold: float = 0.7) -> list[Observation]:
        """Return observations with importance >= *threshold*."""
        return [o for o in self.observations if o.importance >= threshold]

    # ------------------------------------------------------------------
    # Player models
    # ------------------------------------------------------------------

    def get_player_model(self, player_id: str) -> PlayerModel:
        """Return the model for *player_id*, creating a stub if needed."""
        if player_id not in self.player_models:
            self.player_models[player_id] = PlayerModel(
                player_id=player_id,
                name=player_id,
            )
        return self.player_models[player_id]

    def update_player_model(self, player_id: str, **kwargs) -> None:
        """Update fields on the player model for *player_id*.

        Supports any field of :class:`PlayerModel` as a keyword argument.
        List fields (``notes``, ``voted_for``, ``voted_by``) are *appended*
        to rather than replaced when the value is a single string.
        """
        model = self.get_player_model(player_id)

        for key, value in kwargs.items():
            if not hasattr(model, key):
                logger.warning("PlayerModel has no attribute %r, skipping", key)
                continue

            current = getattr(model, key)
            # For list fields, append a single value instead of replacing.
            if isinstance(current, list) and isinstance(value, str):
                current.append(value)
            else:
                setattr(model, key, value)

        logger.debug("Player model updated: %s %s", player_id, kwargs)

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def add_decision(self, decision_dict: dict) -> None:
        """Record a decision the agent made (for later analysis)."""
        self.decisions.append(decision_dict)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def extract_cross_game_learnings(self, game_number: int) -> list[str]:
        """Extract key learnings from this game for cross-game memory.

        Returns a list of summary strings about player behaviour.
        """
        learnings: list[str] = []
        for pm in self.player_models.values():
            parts = []
            if pm.notes:
                parts.append("; ".join(pm.notes[-3:]))
            if pm.claimed_role:
                parts.append(f"claimed to be {pm.claimed_role}")
            if pm.suspicion != 0.5:
                parts.append(f"suspicion={pm.suspicion:.1f}")
            if parts:
                learnings.append(f"Game {game_number}: {pm.name} -- {', '.join(parts)}")
        return learnings

    def inject_cross_game_memories(self, learnings: list[str]) -> None:
        """Inject learnings from previous games as observations."""
        if not learnings:
            return
        for learning in learnings[-10:]:  # keep at most 10 recent learnings
            self.observations.insert(
                0,
                Observation(
                    day=0,
                    phase="PREGAME",
                    content=f"[Prior game memory] {learning}",
                    importance=0.6,
                    source="cross_game",
                ),
            )

    def summarize_for_prompt(self) -> str:
        """Return a formatted summary of key memories for LLM context.

        The summary includes important observations and the current state
        of all player models.
        """
        lines: list[str] = []

        # --- Important observations ---
        important = self.get_important_observations(threshold=0.7)
        if important:
            lines.append("=== Key Observations ===")
            for obs in important:
                lines.append(f"- [Day {obs.day}, {obs.phase}] {obs.content}")
            lines.append("")

        # --- Recent observations (that are not already listed) ---
        recent = self.get_recent_observations(n=10)
        important_set = set(id(o) for o in important)
        recent_only = [o for o in recent if id(o) not in important_set]
        if recent_only:
            lines.append("=== Recent Observations ===")
            for obs in recent_only:
                lines.append(f"- [Day {obs.day}, {obs.phase}] {obs.content}")
            lines.append("")

        # --- Player models ---
        if self.player_models:
            lines.append("=== Player Assessments ===")
            for pm in self.player_models.values():
                parts = [f"{pm.name} (suspicion={pm.suspicion:.1f}, trust={pm.trust:.1f})"]
                if pm.claimed_role:
                    parts.append(f"claims to be {pm.claimed_role}")
                if pm.notes:
                    parts.append(f"notes: {'; '.join(pm.notes[-3:])}")
                lines.append(f"- {', '.join(parts)}")
            lines.append("")

        return "\n".join(lines) if lines else "(no memories yet)"
