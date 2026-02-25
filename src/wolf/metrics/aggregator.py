"""Metrics aggregation across multiple game summaries."""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from typing import Any

from wolf.metrics.scorers.reasoning import ReasoningScorer
from wolf.metrics.scorers.social import SocialScorer
from wolf.metrics.scorers.win_rate import WinRateScorer

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Aggregates game summaries by running all scorers and computing
    cross-game statistics.

    Results are grouped by model for benchmark comparison.
    """

    def __init__(self) -> None:
        self._win_rate_scorer = WinRateScorer()
        self._reasoning_scorer = ReasoningScorer()
        self._social_scorer = SocialScorer()

    def aggregate(
        self, game_summaries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run all scorers and compute cross-game statistics.

        Parameters
        ----------
        game_summaries:
            List of dicts as returned by
            :meth:`MetricsCollector.get_game_summary`.

        Returns
        -------
        dict
            Combined results from all scorers, plus cross-game stats
            and per-model groupings.
        """
        if not game_summaries:
            return {"error": "No game summaries provided"}

        # Run individual scorers
        win_rate_results = self._win_rate_scorer.score(game_summaries)
        reasoning_results = self._reasoning_scorer.score(game_summaries)
        social_results = self._social_scorer.score(game_summaries)

        # Cross-game statistics
        cross_game = _compute_cross_game_stats(game_summaries)

        # Per-model grouping
        model_comparison = _group_by_model(game_summaries)

        return {
            "num_games": len(game_summaries),
            "win_rates": win_rate_results,
            "reasoning": reasoning_results,
            "social": social_results,
            "cross_game": cross_game,
            "model_comparison": model_comparison,
        }


def _compute_cross_game_stats(
    game_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate statistics across all games.

    Uses standard library ``statistics`` for mean/stdev and manual
    computation for confidence intervals.
    """
    game_lengths: list[int] = []
    total_eliminations: list[int] = []
    total_speeches: list[int] = []

    for summary in game_summaries:
        game_lengths.append(summary.get("total_days", 0))

        elims = 0
        speeches = 0
        for player in summary.get("players", []):
            if not player.get("is_alive", True):
                elims += 1
            speeches += player.get("speeches", 0)

        total_eliminations.append(elims)
        total_speeches.append(speeches)

    return {
        "game_length": _summary_stats(game_lengths),
        "eliminations_per_game": _summary_stats(total_eliminations),
        "speeches_per_game": _summary_stats(total_speeches),
    }


def _summary_stats(values: list[int | float]) -> dict[str, float]:
    """Compute mean, standard deviation, and 95% confidence interval."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n >= 2 else 0.0

    # 95% confidence interval using t-distribution approximation
    # For large n, z ~ 1.96; for small n we use a simple lookup
    if n >= 2:
        # Approximate t-value for 95% CI
        t_value = _t_critical(n - 1)
        margin = t_value * std / math.sqrt(n)
    else:
        margin = 0.0

    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci_lower": round(mean - margin, 3),
        "ci_upper": round(mean + margin, 3),
        "n": n,
    }


def _t_critical(df: int) -> float:
    """Approximate t-critical value for 95% CI (two-tailed).

    Uses a simple lookup for small df and falls back to 1.96 for large df.
    """
    # Selected t-values for 95% CI (two-tailed, alpha=0.05)
    table: dict[int, float] = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        25: 2.060,
        30: 2.042,
        40: 2.021,
        50: 2.009,
        60: 2.000,
        80: 1.990,
        100: 1.984,
        120: 1.980,
    }

    if df in table:
        return table[df]

    # Find nearest lower key
    keys = sorted(table.keys())
    for i in range(len(keys) - 1, -1, -1):
        if keys[i] <= df:
            return table[keys[i]]

    # Fallback for large df
    return 1.96


def _group_by_model(
    game_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Group player statistics by model for benchmark comparison."""
    model_stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for summary in game_summaries:
        result = summary.get("result", {})
        winners = set(result.get("winners", []))

        for player in summary.get("players", []):
            model = player.get("model", "")
            if not model:
                model = "unknown"

            pid = player.get("player_id", "")
            won = 1.0 if pid in winners else 0.0
            survived = 1.0 if player.get("is_alive", False) else 0.0
            survival_time = float(player.get("survived_until", 0))
            speeches = float(player.get("speeches", 0))

            model_stats[model]["wins"].append(won)
            model_stats[model]["survival"].append(survived)
            model_stats[model]["survival_time"].append(survival_time)
            model_stats[model]["speeches"].append(speeches)

    result_by_model: dict[str, Any] = {}
    for model, stats in model_stats.items():
        result_by_model[model] = {
            metric: _summary_stats(values)
            for metric, values in stats.items()
        }

    return result_by_model
