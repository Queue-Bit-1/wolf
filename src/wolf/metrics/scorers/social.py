"""Social dynamics scorer -- persuasion and voting bloc analysis."""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Any

logger = logging.getLogger(__name__)


class SocialScorer:
    """Evaluates social dynamics across games.

    Computes:
    - ``persuasion_effectiveness``: did other players follow a player's
      vote suggestions?
    - ``voting_bloc_detection``: players who consistently voted together.
    """

    def score(self, game_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        """Score social dynamics across a batch of games.

        Parameters
        ----------
        game_summaries:
            List of dicts as returned by
            :meth:`MetricsCollector.get_game_summary`.

        Returns
        -------
        dict
            Social dynamics metrics.
        """
        persuasion: dict[str, dict[str, int]] = defaultdict(
            lambda: {"followed": 0, "total": 0}
        )
        # Track pairwise voting agreement across all games
        pair_agreement: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"agree": 0, "total": 0}
        )

        for summary in game_summaries:
            players = summary.get("players", [])

            # Build per-player vote sequences for this game
            player_votes: dict[str, list[str | None]] = {}
            player_speeches: dict[str, list[str]] = {}

            for player in players:
                pid = player.get("player_id", "")
                player_votes[pid] = player.get("vote_targets", [])
                player_speeches[pid] = player.get("speech_contents", [])

            # --- Persuasion effectiveness ---
            # A player is "persuasive" if, after they speak about voting
            # for a target, other players subsequently vote for that same
            # target in the same round.
            _compute_persuasion(
                players, player_votes, player_speeches, persuasion
            )

            # --- Voting bloc detection ---
            # For each pair of players, count how often they voted for the
            # same target in the same round.
            _compute_pairwise_agreement(player_votes, pair_agreement)

        # Build final persuasion scores
        persuasion_scores: dict[str, float] = {}
        for pid, data in persuasion.items():
            if data["total"] > 0:
                persuasion_scores[pid] = data["followed"] / data["total"]
            else:
                persuasion_scores[pid] = 0.0

        # Build voting blocs: pairs with agreement rate > 0.6
        voting_blocs: list[dict[str, Any]] = []
        for (p1, p2), data in pair_agreement.items():
            if data["total"] >= 2:
                rate = data["agree"] / data["total"]
                if rate > 0.6:
                    voting_blocs.append(
                        {
                            "players": [p1, p2],
                            "agreement_rate": round(rate, 3),
                            "rounds_together": data["total"],
                        }
                    )

        # Sort by agreement rate descending
        voting_blocs.sort(key=lambda b: b["agreement_rate"], reverse=True)

        return {
            "persuasion_effectiveness": persuasion_scores,
            "voting_bloc_detection": voting_blocs,
        }


def _compute_persuasion(
    players: list[dict[str, Any]],
    player_votes: dict[str, list[str | None]],
    player_speeches: dict[str, list[str]],
    results: dict[str, dict[str, int]],
) -> None:
    """Estimate persuasion by checking if mentioned targets get voted for.

    For each player, we look at the targets they mentioned in speeches.
    If another player voted for that same target, it counts as a
    "followed" suggestion.
    """
    all_player_ids = [p.get("player_id", "") for p in players]

    for player in players:
        pid = player.get("player_id", "")
        speeches = player_speeches.get(pid, [])
        vote_targets = player_votes.get(pid, [])

        if not speeches:
            continue

        # Find player IDs mentioned in speeches
        speech_text = " ".join(speeches).lower()
        mentioned_targets: set[str] = set()
        for other_pid in all_player_ids:
            if other_pid != pid and other_pid.lower() in speech_text:
                mentioned_targets.add(other_pid)

        if not mentioned_targets:
            continue

        # Check if the player voted for any of the mentioned targets
        actual_vote_set = {t for t in vote_targets if t is not None}

        # For each mentioned target the player voted for, see if
        # others also voted for it
        suggested_targets = mentioned_targets & actual_vote_set

        for target in suggested_targets:
            results[pid]["total"] += 1
            # Count how many other players voted for the same target
            followers = 0
            for other_pid in all_player_ids:
                if other_pid == pid:
                    continue
                other_votes = player_votes.get(other_pid, [])
                if target in other_votes:
                    followers += 1
            if followers > 0:
                results[pid]["followed"] += 1


def _compute_pairwise_agreement(
    player_votes: dict[str, list[str | None]],
    results: dict[tuple[str, str], dict[str, int]],
) -> None:
    """Compute pairwise voting agreement between all player pairs."""
    player_ids = sorted(player_votes.keys())

    for p1, p2 in combinations(player_ids, 2):
        votes1 = player_votes[p1]
        votes2 = player_votes[p2]

        # Compare votes round-by-round (aligned by index)
        rounds = min(len(votes1), len(votes2))
        for i in range(rounds):
            v1 = votes1[i]
            v2 = votes2[i]
            if v1 is not None and v2 is not None:
                results[(p1, p2)]["total"] += 1
                if v1 == v2:
                    results[(p1, p2)]["agree"] += 1
