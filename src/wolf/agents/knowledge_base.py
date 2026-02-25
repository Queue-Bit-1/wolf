"""Agent-driven knowledge base: notes and player assessments.

Unlike the old ``AgentMemory`` (which was auto-populated by the engine),
the KnowledgeBase is written to *only* by the agent through explicit tool
calls.  The engine never writes to it -- this ensures agents control what
they remember and prevents information leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PlayerAssessment:
    """An agent's assessment of another player."""

    name: str
    entries: list[str] = field(default_factory=list)


class KnowledgeBase:
    """Persistent memory that an agent manages through tools.

    Persists across phases within a game (conversation context resets,
    but the KB does not).  Provides two storage areas:

    * **Notes** -- a free-form scratchpad for the agent's thoughts.
    * **Assessments** -- per-player assessment entries.

    Cross-game support: ``extract_learnings()`` / ``inject_learnings()``
    can be used between games in a batch for long-term memory.
    """

    def __init__(self, player_name: str) -> None:
        self.player_name = player_name
        self._notes: list[str] = []
        self._assessments: dict[str, PlayerAssessment] = {}
        self._learnings: list[str] = []  # cross-game learnings

    # ------------------------------------------------------------------
    # Notes (scratchpad)
    # ------------------------------------------------------------------

    def read_notes(self) -> str:
        """Return all notes as a single text block."""
        if not self._notes:
            return "(no notes yet)"
        return "\n".join(f"- {note}" for note in self._notes)

    def write_notes(self, text: str) -> str:
        """Append a note entry. Returns confirmation text."""
        text = text.strip()
        if not text:
            return "Error: empty note."
        self._notes.append(text)
        return f"Note saved. ({len(self._notes)} total notes)"

    def clear_notes(self) -> str:
        """Clear all notes. Returns confirmation."""
        count = len(self._notes)
        self._notes.clear()
        return f"Cleared {count} notes."

    # ------------------------------------------------------------------
    # Player assessments
    # ------------------------------------------------------------------

    def assess_player(self, args: str) -> str:
        """Record an assessment of a player.

        Expected format: ``"player_name: assessment text"``
        """
        if ":" not in args:
            return "Error: expected format 'player_name: assessment text'"

        name, text = args.split(":", 1)
        name = name.strip()
        text = text.strip()

        if not name or not text:
            return "Error: both player name and assessment text are required."

        if name not in self._assessments:
            self._assessments[name] = PlayerAssessment(name=name)
        self._assessments[name].entries.append(text)
        count = len(self._assessments[name].entries)
        return f"Assessment of {name} recorded. ({count} entries for {name})"

    def read_assessments(self) -> str:
        """Return all player assessments as a text block."""
        if not self._assessments:
            return "(no assessments yet)"

        lines: list[str] = []
        for name, assessment in sorted(self._assessments.items()):
            lines.append(f"{name}:")
            for entry in assessment.entries:
                lines.append(f"  - {entry}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Briefing summary
    # ------------------------------------------------------------------

    def summarize_for_briefing(self) -> str:
        """Return a compact summary of the KB for inclusion in briefings.

        This is automatically included in each phase's briefing so the
        agent sees their accumulated knowledge without needing to invoke
        ``read_notes`` or ``read_assessments`` first.
        """
        parts: list[str] = []

        if self._notes:
            parts.append("=== Your Notes ===")
            for note in self._notes[-10:]:  # cap at 10 most recent
                parts.append(f"- {note}")
            if len(self._notes) > 10:
                parts.append(f"  ... ({len(self._notes) - 10} earlier notes omitted)")
            parts.append("")

        if self._assessments:
            parts.append("=== Your Player Assessments ===")
            for name, assessment in sorted(self._assessments.items()):
                # Show last 3 entries per player
                recent = assessment.entries[-3:]
                parts.append(f"{name}: {'; '.join(recent)}")
            parts.append("")

        if self._learnings:
            parts.append("=== Lessons from Prior Games ===")
            for learning in self._learnings[-5:]:
                parts.append(f"- {learning}")
            parts.append("")

        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Cross-game persistence
    # ------------------------------------------------------------------

    def extract_learnings(self, game_number: int) -> list[str]:
        """Extract key learnings from this game for future games.

        Returns a list of summary strings.
        """
        learnings: list[str] = []

        # Include latest notes as potential learnings
        if self._notes:
            summary = "; ".join(self._notes[-5:])
            learnings.append(f"Game {game_number} notes: {summary}")

        # Include player assessment summaries
        for name, assessment in self._assessments.items():
            if assessment.entries:
                recent = "; ".join(assessment.entries[-2:])
                learnings.append(
                    f"Game {game_number}: {name} -- {recent}"
                )

        return learnings

    def inject_learnings(self, learnings: list[str]) -> None:
        """Inject learnings from prior games."""
        self._learnings = list(learnings[-10:])  # cap at 10
