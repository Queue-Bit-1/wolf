"""LLM-based judge for qualitative game evaluation."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import openai

from wolf.config.schema import ModelConfig

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are a Werewolf game analyst. You will be given a summary of a completed
Werewolf game and must evaluate each player's performance.

For each player, score the following dimensions on a scale of 1-10:

- **strategic_depth**: How well did the player plan ahead and adapt their strategy?
- **persuasion_quality**: How effectively did the player persuade others?
- **deception_skill**: How well did the player conceal their true role or detect deception?
- **logical_coherence**: How logically consistent were the player's arguments and actions?

Return your evaluation as a JSON object with this exact structure:
{
  "evaluations": {
    "<player_id>": {
      "strategic_depth": <int 1-10>,
      "persuasion_quality": <int 1-10>,
      "deception_skill": <int 1-10>,
      "logical_coherence": <int 1-10>,
      "comments": "<brief comment>"
    }
  }
}

Return ONLY the JSON object, no additional text.
"""


def _format_game_for_judge(game_summary: dict[str, Any]) -> str:
    """Format a game summary into a human-readable prompt for the judge."""
    lines: list[str] = []
    result = game_summary.get("result", {})
    lines.append(f"Game Result: {result.get('winning_team', 'unknown')} wins")
    lines.append(f"Reason: {result.get('reason', 'N/A')}")
    lines.append(f"Total Days: {game_summary.get('total_days', 0)}")
    lines.append("")

    lines.append("Players:")
    for player in game_summary.get("players", []):
        pid = player.get("player_id", "")
        name = player.get("name", pid)
        role = player.get("role", "unknown")
        team = player.get("team", "unknown")
        alive = "alive" if player.get("is_alive", False) else "eliminated"
        survived = player.get("survived_until", 0)
        speeches = player.get("speeches", 0)
        votes = player.get("votes_cast", 0)

        lines.append(
            f"  - {name} ({pid}): role={role}, team={team}, "
            f"status={alive}, survived_until=day_{survived}, "
            f"speeches={speeches}, votes_cast={votes}"
        )

        # Include speech excerpts (first 200 chars each, up to 3)
        speech_contents = player.get("speech_contents", [])
        for i, speech in enumerate(speech_contents[:3]):
            excerpt = speech[:200] + "..." if len(speech) > 200 else speech
            lines.append(f"    Speech {i + 1}: {excerpt}")

    return "\n".join(lines)


class LLMJudge:
    """Uses an LLM to evaluate gameplay quality.

    Constructor takes a :class:`ModelConfig` for the judge model endpoint.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=30.0),
        )
        self._client = openai.AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
            http_client=self._http_client,
        )

    async def evaluate_game(
        self, game_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Use the LLM to evaluate player performance in a game.

        Parameters
        ----------
        game_summary:
            Game summary dict as returned by
            :meth:`MetricsCollector.get_game_summary`.

        Returns
        -------
        dict
            Per-player evaluations with scores for ``strategic_depth``,
            ``persuasion_quality``, ``deception_skill``, and
            ``logical_coherence``.
        """
        game_text = _format_game_for_judge(game_summary)

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": game_text},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.action_temperature,
                max_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content or ""
            evaluations = _parse_judge_response(content)
            return evaluations

        except Exception:
            logger.exception("LLM judge evaluation failed")
            return _default_evaluations(game_summary)

    async def close(self) -> None:
        """Shut down the underlying HTTP transport."""
        await self._http_client.aclose()


def _parse_judge_response(content: str) -> dict[str, Any]:
    """Parse the judge LLM response into structured evaluations."""
    # Try to extract JSON from the response
    content = content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (code block markers)
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            if line.strip() == "```" and in_block:
                break
            if in_block:
                json_lines.append(line)
        content = "\n".join(json_lines)

    try:
        data = json.loads(content)
        if "evaluations" in data:
            return data["evaluations"]
        return data
    except json.JSONDecodeError:
        logger.warning("Could not parse judge response as JSON")
        return {}


def _default_evaluations(game_summary: dict[str, Any]) -> dict[str, Any]:
    """Return default (neutral) evaluations when the judge fails."""
    evaluations: dict[str, Any] = {}
    for player in game_summary.get("players", []):
        pid = player.get("player_id", "")
        evaluations[pid] = {
            "strategic_depth": 5,
            "persuasion_quality": 5,
            "deception_skill": 5,
            "logical_coherence": 5,
            "comments": "Judge evaluation unavailable.",
        }
    return evaluations
