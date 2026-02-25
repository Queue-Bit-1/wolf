"""Reasoning quality scorer -- evaluates logical play quality."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class ReasoningScorer:
    """Evaluates reasoning quality across games.

    Computes:
    - ``accusation_accuracy``: how often accusations targeted actual wolves.
    - ``vote_consistency``: how often votes matched stated suspicions.
    - ``deception_success``: for wolves, how often they avoided being voted out.
    """

    def score(self, game_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        """Score reasoning quality across a batch of games.

        Parameters
        ----------
        game_summaries:
            List of dicts as returned by
            :meth:`MetricsCollector.get_game_summary`.

        Returns
        -------
        dict
            Reasoning quality metrics keyed by player_id.
        """
        accusation_accuracy: dict[str, dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )
        vote_consistency: dict[str, dict[str, int]] = defaultdict(
            lambda: {"consistent": 0, "total": 0}
        )
        deception_success: dict[str, dict[str, int]] = defaultdict(
            lambda: {"not_voted_out": 0, "total_games": 0}
        )

        for summary in game_summaries:
            players = summary.get("players", [])

            # Build a lookup: player_id -> role/team
            role_map: dict[str, str] = {}
            team_map: dict[str, str] = {}
            for p in players:
                pid = p.get("player_id", "")
                role_map[pid] = p.get("role", "")
                team_map[pid] = p.get("team", "")

            wolf_ids = {
                pid for pid, team in team_map.items() if team == "werewolf"
            }

            for player in players:
                pid = player.get("player_id", "")
                team = player.get("team", "")
                vote_targets = player.get("vote_targets", [])
                speeches = player.get("speech_contents", [])
                elimination_cause = player.get("elimination_cause", "")

                # --- Accusation accuracy ---
                # We treat vote targets as "accusations" for village-team
                # players: did they vote for actual wolves?
                if team != "werewolf":
                    for target in vote_targets:
                        if target is not None:
                            accusation_accuracy[pid]["total"] += 1
                            if target in wolf_ids:
                                accusation_accuracy[pid]["correct"] += 1

                # --- Vote consistency ---
                # Check if a player's votes align with the suspicious
                # players they mention in their speeches. We look for
                # player IDs or names mentioned in the speech preceding
                # each vote.
                _compute_vote_consistency(
                    pid, speeches, vote_targets, vote_consistency
                )

                # --- Deception success (wolves only) ---
                if team == "werewolf":
                    deception_success[pid]["total_games"] += 1
                    if elimination_cause != "vote":
                        deception_success[pid]["not_voted_out"] += 1

        return {
            "accusation_accuracy": {
                pid: (
                    data["correct"] / data["total"]
                    if data["total"] > 0
                    else 0.0
                )
                for pid, data in accusation_accuracy.items()
            },
            "vote_consistency": {
                pid: (
                    data["consistent"] / data["total"]
                    if data["total"] > 0
                    else 0.0
                )
                for pid, data in vote_consistency.items()
            },
            "deception_success": {
                pid: (
                    data["not_voted_out"] / data["total_games"]
                    if data["total_games"] > 0
                    else 0.0
                )
                for pid, data in deception_success.items()
            },
        }


def _compute_vote_consistency(
    player_id: str,
    speeches: list[str],
    vote_targets: list[str | None],
    results: dict[str, dict[str, int]],
) -> None:
    """Check whether votes are consistent with speech mentions.

    A vote is considered "consistent" if the voted target was mentioned
    (by player_id substring) in the player's speeches within the same
    game. This is a heuristic approximation.
    """
    # Concatenate all speeches into one search corpus
    all_speech_text = " ".join(speeches).lower()

    for target in vote_targets:
        if target is not None:
            results[player_id]["total"] += 1
            # Simple heuristic: was the target ID mentioned in any speech?
            if target.lower() in all_speech_text:
                results[player_id]["consistent"] += 1
