"""JSON exporter for metrics results."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class JSONExporter:
    """Exports metrics results to a pretty-printed JSON file."""

    def export(self, results: dict[str, Any], output_path: str) -> None:
        """Write results to a JSON file with pretty formatting.

        Parameters
        ----------
        results:
            The metrics results dict to export.
        output_path:
            File path for the output JSON file.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        logger.info("JSON results exported to %s", output_path)
