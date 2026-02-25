"""Token usage tracker for monitoring LLM consumption."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _PlayerUsage:
    """Accumulated token counts for one player."""

    total_input: int = 0
    total_output: int = 0
    by_call_type: dict[str, dict[str, int]] = field(default_factory=dict)


class TokenTracker:
    """Tracks per-player and aggregate token usage across a game.

    Call :meth:`record` after each LLM call.  Use :meth:`get_player_usage`
    or :meth:`get_total_usage` to retrieve summaries.
    """

    def __init__(self) -> None:
        self._usage: dict[str, _PlayerUsage] = defaultdict(_PlayerUsage)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        player_id: str,
        input_tokens: int,
        output_tokens: int,
        call_type: str = "reasoning",
    ) -> None:
        """Record token usage for a single LLM call.

        Parameters
        ----------
        player_id:
            Identifier of the player who made the call.
        input_tokens:
            Number of prompt / input tokens.
        output_tokens:
            Number of completion / output tokens.
        call_type:
            Category of the call, typically ``"reasoning"`` or ``"action"``.
        """
        usage = self._usage[player_id]
        usage.total_input += input_tokens
        usage.total_output += output_tokens

        if call_type not in usage.by_call_type:
            usage.by_call_type[call_type] = {"input": 0, "output": 0}
        usage.by_call_type[call_type]["input"] += input_tokens
        usage.by_call_type[call_type]["output"] += output_tokens

        logger.debug(
            "Tokens recorded: player=%s type=%s in=%d out=%d",
            player_id,
            call_type,
            input_tokens,
            output_tokens,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_player_usage(self, player_id: str) -> dict:
        """Return a usage summary for a single player.

        Returns a dict with keys ``total_input``, ``total_output``, and
        ``by_call_type`` (mapping call-type names to ``{input, output}``
        dicts).
        """
        usage = self._usage.get(player_id)
        if usage is None:
            return {
                "total_input": 0,
                "total_output": 0,
                "by_call_type": {},
            }
        return {
            "total_input": usage.total_input,
            "total_output": usage.total_output,
            "by_call_type": dict(usage.by_call_type),
        }

    def get_total_usage(self) -> dict:
        """Return aggregate usage across all players.

        Returns a dict with ``total_input``, ``total_output``, and
        ``per_player`` breakdown.
        """
        total_input = 0
        total_output = 0
        per_player: dict[str, dict] = {}

        for player_id, usage in self._usage.items():
            total_input += usage.total_input
            total_output += usage.total_output
            per_player[player_id] = {
                "total_input": usage.total_input,
                "total_output": usage.total_output,
                "by_call_type": dict(usage.by_call_type),
            }

        return {
            "total_input": total_input,
            "total_output": total_output,
            "per_player": per_player,
        }

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded usage data."""
        self._usage.clear()
        logger.debug("Token tracker reset")
