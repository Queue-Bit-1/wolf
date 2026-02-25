"""CSV exporter for metrics results."""

from __future__ import annotations

import csv
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class CSVExporter:
    """Exports metrics results to a flattened CSV file."""

    def export(self, results: dict[str, Any], output_path: str) -> None:
        """Flatten results into CSV rows and write to file.

        Parameters
        ----------
        results:
            The metrics results dict to export.
        output_path:
            File path for the output CSV file.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        rows = _flatten_dict(results)

        if not rows:
            logger.warning("No data to export to CSV")
            return

        # All rows share the same keys
        fieldnames = list(rows[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("CSV results exported to %s (%d rows)", output_path, len(rows))


def _flatten_dict(
    data: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> list[dict[str, str]]:
    """Flatten a nested dict into a list of flat row dicts.

    Strategy: walk the nested dict, collecting leaf values into a single
    flat row. If we encounter lists of dicts (e.g., player stats), each
    list item becomes a separate row.
    """
    # First, check if there are player-level or game-level lists we can
    # iterate as rows.
    rows = _extract_rows(data)
    if rows:
        return rows

    # Otherwise flatten the whole thing into a single row
    flat: dict[str, str] = {}
    _flatten_recursive(data, parent_key, sep, flat)
    if flat:
        return [flat]
    return []


def _extract_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    """Try to extract meaningful rows from the data structure.

    Looks for model_comparison or similar structures that map to rows.
    """
    rows: list[dict[str, str]] = []

    # Model comparison: one row per model
    model_comparison = data.get("model_comparison", {})
    if model_comparison and isinstance(model_comparison, dict):
        for model, stats in model_comparison.items():
            row: dict[str, str] = {"model": str(model)}
            if isinstance(stats, dict):
                for metric, values in stats.items():
                    if isinstance(values, dict):
                        for stat_name, stat_val in values.items():
                            key = f"{metric}.{stat_name}"
                            row[key] = str(stat_val)
                    else:
                        row[metric] = str(values)
            rows.append(row)
        return rows

    # Win rates by team/role: flatten into rows
    win_rates = data.get("win_rates", {})
    if win_rates and isinstance(win_rates, dict):
        # Team win rates
        team_rates = win_rates.get("win_rate_by_team", {})
        role_rates = win_rates.get("win_rate_by_role", {})
        survival_rates = win_rates.get("survival_rate_by_role", {})
        avg_survival = win_rates.get("avg_survival_time_by_role", {})

        if role_rates:
            for role, rate in role_rates.items():
                row = {
                    "role": str(role),
                    "win_rate": str(rate),
                    "survival_rate": str(survival_rates.get(role, "")),
                    "avg_survival_time": str(avg_survival.get(role, "")),
                }
                rows.append(row)
            return rows

        if team_rates:
            for team, rate in team_rates.items():
                rows.append({"team": str(team), "win_rate": str(rate)})
            return rows

    # Generic fallback: single flattened row
    return []


def _flatten_recursive(
    data: Any,
    parent_key: str,
    sep: str,
    result: dict[str, str],
) -> None:
    """Recursively flatten a nested structure into a flat dict."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            _flatten_recursive(value, new_key, sep, result)
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            _flatten_recursive(item, new_key, sep, result)
    else:
        result[parent_key] = str(data)
