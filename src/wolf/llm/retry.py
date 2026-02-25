"""Retry helper with exponential back-off for LLM API calls."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, TypeVar

import openai

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that are safe to retry.
_RETRYABLE = (
    openai.APIError,
    openai.APITimeoutError,
    openai.RateLimitError,
    ConnectionError,
)


async def retry_with_backoff(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> T:
    """Call *fn* with retries on transient failures.

    Uses exponential back-off with jitter::

        delay = min(base_delay * 2^attempt, max_delay) * uniform(0.5, 1.0)

    Parameters
    ----------
    fn:
        An async callable to invoke.
    max_attempts:
        Total number of tries (including the first).
    base_delay:
        Starting delay in seconds.
    max_delay:
        Cap on the computed delay.
    *args, **kwargs:
        Forwarded to *fn*.

    Returns
    -------
    The return value of *fn* on the first successful call.

    Raises
    ------
    Exception
        The last exception if all attempts fail.
    """
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await fn(*args, **kwargs)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                logger.error(
                    "All %d retry attempts exhausted for %s: %s",
                    max_attempts,
                    fn.__qualname__,
                    exc,
                )
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0.5, 1.0)
            sleep_time = delay * jitter

            logger.warning(
                "Retry %d/%d for %s after %.1fs (error: %s)",
                attempt + 1,
                max_attempts,
                fn.__qualname__,
                sleep_time,
                exc,
            )
            await asyncio.sleep(sleep_time)

    # Should never reach here, but satisfy type checkers.
    assert last_exc is not None
    raise last_exc
