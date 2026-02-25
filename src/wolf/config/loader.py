"""Configuration loading, merging, and model discovery."""

from __future__ import annotations

import copy
import logging
from typing import Any

import yaml

from wolf.config.schema import GameConfig

logger = logging.getLogger(__name__)


def load_config(path: str | None = None) -> GameConfig:
    """Load a game configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML configuration file.  If *None* or the file does
        not exist, a default :class:`GameConfig` is returned.

    Returns
    -------
    GameConfig
        Parsed and validated configuration.
    """
    if path is None:
        logger.debug("No config path provided, using defaults")
        return GameConfig()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config file not found: %s -- using defaults", path)
        return GameConfig()
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML config %s: %s", path, exc)
        return GameConfig()

    if not isinstance(data, dict):
        logger.warning("Config file %s did not produce a dict, using defaults", path)
        return GameConfig()

    try:
        return GameConfig.model_validate(data)
    except Exception as exc:
        logger.error("Config validation failed for %s: %s", path, exc)
        return GameConfig()


def merge_configs(base: GameConfig, overrides: dict[str, Any]) -> GameConfig:
    """Deep-merge an override dict into a base config.

    Parameters
    ----------
    base:
        The base game configuration.
    overrides:
        A (possibly nested) dict of values to override.

    Returns
    -------
    GameConfig
        A new configuration with overrides applied.
    """
    base_dict = base.model_dump()
    merged = _deep_merge(base_dict, overrides)

    try:
        return GameConfig.model_validate(merged)
    except Exception as exc:
        logger.error("Merged config validation failed: %s", exc)
        return base


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


async def detect_ollama_models(
    api_base: str = "http://localhost:11434",
) -> list[str]:
    """Discover available models from an Ollama instance.

    Calls ``GET {api_base}/api/tags`` to retrieve the model list.

    Parameters
    ----------
    api_base:
        Base URL for the Ollama API.

    Returns
    -------
    list[str]
        Names of available models, or an empty list on failure.
    """
    import httpx

    url = f"{api_base}/api/tags"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            models: list[str] = []
            for model_entry in data.get("models", []):
                name = model_entry.get("name", "")
                if name:
                    models.append(name)

            logger.info(
                "Discovered %d Ollama model(s) at %s", len(models), api_base
            )
            return models

    except httpx.ConnectError:
        logger.debug("Cannot connect to Ollama at %s", api_base)
        return []
    except Exception:
        logger.debug("Failed to detect Ollama models at %s", api_base, exc_info=True)
        return []
