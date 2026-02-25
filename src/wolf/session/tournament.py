"""Tournament runner for round-robin model matchups."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from wolf.config.schema import GameConfig
from wolf.metrics.aggregator import MetricsAggregator
from wolf.session.batch import BatchRunner
from wolf.session.runner import GameResult

logger = logging.getLogger(__name__)


@dataclass
class TournamentResult:
    """Aggregated results from a tournament run."""

    results: list[GameResult]
    aggregated_metrics: dict[str, Any]
    model_rankings: list[dict[str, Any]]


class TournamentRunner:
    """Runs a round-robin tournament across multiple configurations.

    Each configuration represents a different model matchup. The
    tournament runs all configurations and aggregates the results.
    """

    def __init__(self, configs: list[GameConfig]) -> None:
        self.configs = configs

    async def run(self) -> TournamentResult:
        """Execute the tournament and return aggregated results.

        Runs all game configurations, collects results, aggregates
        metrics, and produces model rankings.

        Returns
        -------
        TournamentResult
            Combined results with aggregated metrics and rankings.
        """
        all_results: list[GameResult] = []

        logger.info(
            "TournamentRunner: starting with %d config(s)", len(self.configs)
        )

        for i, config in enumerate(self.configs):
            logger.info(
                "Tournament config %d / %d: %s (%d games)",
                i + 1,
                len(self.configs),
                config.game_name,
                config.benchmark.num_games,
            )

            batch = BatchRunner(config)
            results = await batch.run(
                num_games=config.benchmark.num_games,
                parallel=config.benchmark.parallel_games,
            )
            all_results.extend(results)

        # Aggregate all game summaries
        summaries = [r.game_summary for r in all_results]
        aggregator = MetricsAggregator()
        aggregated = aggregator.aggregate(summaries)

        # Compute model rankings
        rankings = _compute_rankings(aggregated)

        logger.info(
            "TournamentRunner: completed %d total games", len(all_results)
        )

        return TournamentResult(
            results=all_results,
            aggregated_metrics=aggregated,
            model_rankings=rankings,
        )


def _compute_rankings(
    aggregated: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank models by composite score from aggregated metrics.

    The composite score is a weighted average of win rate, survival rate,
    and average survival time (all normalised to 0-1 range).
    """
    model_comparison = aggregated.get("model_comparison", {})
    if not model_comparison:
        return []

    rankings: list[dict[str, Any]] = []

    for model, stats in model_comparison.items():
        win_mean = _safe_get_mean(stats, "wins")
        survival_mean = _safe_get_mean(stats, "survival")

        # Composite score: weighted sum
        composite = 0.5 * win_mean + 0.3 * survival_mean + 0.2 * _safe_get_mean(stats, "speeches")

        rankings.append(
            {
                "model": model,
                "win_rate": round(win_mean, 4),
                "survival_rate": round(survival_mean, 4),
                "composite_score": round(composite, 4),
                "games_played": _safe_get_n(stats, "wins"),
            }
        )

    # Sort by composite score descending
    rankings.sort(key=lambda r: r["composite_score"], reverse=True)

    # Add rank
    for i, entry in enumerate(rankings):
        entry["rank"] = i + 1

    return rankings


def _safe_get_mean(stats: dict[str, Any], metric: str) -> float:
    """Safely extract the mean from a metric's stats dict."""
    metric_data = stats.get(metric, {})
    if isinstance(metric_data, dict):
        return float(metric_data.get("mean", 0.0))
    return 0.0


def _safe_get_n(stats: dict[str, Any], metric: str) -> int:
    """Safely extract the sample size from a metric's stats dict."""
    metric_data = stats.get(metric, {})
    if isinstance(metric_data, dict):
        return int(metric_data.get("n", 0))
    return 0
