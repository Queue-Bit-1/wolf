"""Win-rate and survival statistics scorer."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class WinRateScorer:
    """Computes win-rate and survival statistics across multiple games.

    Analyses game summaries to produce:
    - ``win_rate_by_team``
    - ``win_rate_by_role``
    - ``survival_rate_by_role``
    - ``avg_survival_time_by_role``
    - ``win_rate_by_model``
    """

    def score(self, game_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        """Score a batch of game summaries.

        Parameters
        ----------
        game_summaries:
            List of dicts as returned by
            :meth:`MetricsCollector.get_game_summary`.

        Returns
        -------
        dict
            Aggregated win-rate and survival metrics.
        """
        # Accumulators
        team_wins: dict[str, int] = defaultdict(int)
        team_games: dict[str, int] = defaultdict(int)

        role_wins: dict[str, int] = defaultdict(int)
        role_games: dict[str, int] = defaultdict(int)

        role_survivals: dict[str, int] = defaultdict(int)
        role_total: dict[str, int] = defaultdict(int)

        role_survival_time: dict[str, list[int]] = defaultdict(list)

        model_wins: dict[str, int] = defaultdict(int)
        model_games: dict[str, int] = defaultdict(int)

        for summary in game_summaries:
            result = summary.get("result", {})
            winning_team = result.get("winning_team", "")
            winners = set(result.get("winners", []))

            # Track which teams participated in this game
            teams_in_game: set[str] = set()

            for player in summary.get("players", []):
                team = player.get("team", "")
                role = player.get("role", "")
                player_id = player.get("player_id", "")
                survived = player.get("is_alive", False)
                survived_until = player.get("survived_until", 0)

                if team:
                    teams_in_game.add(team)

                # Role stats
                if role:
                    role_games[role] += 1
                    role_total[role] += 1
                    role_survival_time[role].append(survived_until)
                    if survived:
                        role_survivals[role] += 1
                    if player_id in winners:
                        role_wins[role] += 1

                # Model stats (from metadata if available)
                model = player.get("model", "")
                if model:
                    model_games[model] += 1
                    if player_id in winners:
                        model_wins[model] += 1

            # Team-level win tracking: each team present counts as one game
            for team in teams_in_game:
                team_games[team] += 1
                if team == winning_team:
                    team_wins[team] += 1

        return {
            "win_rate_by_team": {
                team: team_wins[team] / team_games[team]
                if team_games[team] > 0
                else 0.0
                for team in team_games
            },
            "win_rate_by_role": {
                role: role_wins[role] / role_games[role]
                if role_games[role] > 0
                else 0.0
                for role in role_games
            },
            "survival_rate_by_role": {
                role: role_survivals[role] / role_total[role]
                if role_total[role] > 0
                else 0.0
                for role in role_total
            },
            "avg_survival_time_by_role": {
                role: (
                    sum(times) / len(times) if times else 0.0
                )
                for role, times in role_survival_time.items()
            },
            "win_rate_by_model": {
                model: model_wins[model] / model_games[model]
                if model_games[model] > 0
                else 0.0
                for model in model_games
            },
        }
