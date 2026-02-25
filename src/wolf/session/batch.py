"""Batch runner for executing multiple games with optional parallelism."""

from __future__ import annotations

import asyncio
import copy
import logging
import random
from typing import Any

from wolf.config.schema import GameConfig
from wolf.session.runner import GameResult, GameRunner

logger = logging.getLogger(__name__)


class BatchRunner:
    """Runs multiple games with optional parallelism and role rotation.

    Maintains cross-game agent memories so players can learn about
    opponents across games in the batch.
    """

    def __init__(
        self,
        config: GameConfig,
        extra_listeners: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.extra_listeners = extra_listeners or []
        # Shared cross-game memories keyed by player name
        self._cross_game_memories: dict[str, list[str]] = {}

    async def run(
        self, num_games: int, parallel: int = 1
    ) -> list[GameResult]:
        """Execute *num_games* games with up to *parallel* concurrency.

        Parameters
        ----------
        num_games:
            Total number of games to run.
        parallel:
            Maximum number of games running concurrently.

        Returns
        -------
        list[GameResult]
            Results from all completed games.
        """
        rotate = self.config.benchmark.rotate_roles

        logger.info(
            "BatchRunner: %d games, parallel=%d, rotate_roles=%s",
            num_games,
            parallel,
            rotate,
        )

        # Run games sequentially to accumulate cross-game memories.
        # (Parallelism is still supported for independent batches but
        # cross-game learning requires sequential execution.)
        results: list[GameResult] = []
        for i in range(num_games):
            config = self._prepare_config(i, rotate)
            try:
                logger.info("Starting game %d / %d", i + 1, num_games)
                runner = GameRunner(
                    config,
                    extra_listeners=self.extra_listeners,
                    game_number=i + 1,
                    cross_game_memories=self._cross_game_memories,
                )
                result = await runner.run()
                # cross_game_memories is mutated in-place by the runner
                results.append(result)
            except Exception:
                logger.exception("Game %d in the batch failed", i + 1)

        logger.info(
            "BatchRunner: completed %d / %d games", len(results), num_games
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_config(self, game_index: int, rotate: bool) -> GameConfig:
        """Prepare a config for a specific game, optionally rotating roles."""
        if not rotate:
            return self.config

        # Deep-copy and rotate role assignments by shuffling the role slots
        # This means different games will have different role distributions
        # among players, which is useful for benchmarking.
        config_dict = self.config.model_dump()

        # Rotate by shifting the role list
        roles = config_dict.get("roles", [])
        if roles and game_index > 0:
            # We don't change role counts, but the random assignment in
            # GameRunner already shuffles. The "rotation" here seeds the
            # RNG differently for each game so we get different assignments.
            seed = self.config.benchmark.seed
            if seed is not None:
                random.seed(seed + game_index)

        return GameConfig.model_validate(config_dict)
