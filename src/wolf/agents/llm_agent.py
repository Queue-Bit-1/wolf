"""LLM-powered Werewolf agent with a multi-round ReAct tool loop."""

from __future__ import annotations

import logging
import re
import time as _time
from typing import TYPE_CHECKING, Any

from wolf.agents.base import AgentBase
from wolf.engine.actions import Action, NoAction
from wolf.engine.events import ReasoningEvent
from wolf.llm.client import LLMClient
from wolf.llm.retry import retry_with_backoff
from wolf.llm.token_tracker import TokenTracker

if TYPE_CHECKING:
    from wolf.agents.toolkit import AgentToolkit
    from wolf.config.schema import ModelConfig
    from wolf.roles.base import RoleBase

logger = logging.getLogger(__name__)

# Regex to find  USE: tool_name(args)  in LLM output.
# Handles multi-line args via DOTALL and allows optional whitespace.
# Captures group 1 = tool name (may include prefixes like "tool." or "assistant:").
_TOOL_CALL_RE = re.compile(
    r"USE:\s*([\w.:]+)\((.*)?\)",
    re.DOTALL,
)

# Fallback: bare tool_name(args) at the start of a line (no USE: prefix).
# Also accepts prefixed names like tool.speak() or assistant:speak().
_BARE_TOOL_RE = re.compile(
    r"(?:^|\n)\s*([\w.:]+)\(([^)]*)\)",
)

# Prefixes that models (e.g. gpt-oss) sometimes prepend to tool names.
_TOOL_NAME_PREFIXES = ("tool.", "assistant:", "assistant.")

# Maximum ReAct rounds before forcing a terminal tool.
_MAX_ROUNDS = 8


class LLMAgent(AgentBase):
    """Agent that uses an LLM with a multi-round ReAct tool loop.

    Each ``run_phase`` call:

    1. Builds a FRESH conversation: ``[system_prompt, user: briefing + tools]``
    2. Loops up to ``_MAX_ROUNDS`` rounds:
       a. Calls LLM
       b. Parses ``USE: tool_name(args)`` from response
       c. If terminal tool -> returns the Action
       d. If non-terminal -> executes tool, feeds result back as user message
       e. If no tool found -> reminds agent to use a tool
    3. If max rounds reached -> returns NoAction

    The conversation is fresh per phase -- no history carries over.
    Only the KnowledgeBase persists across phases.
    """

    def __init__(
        self,
        player_id: str,
        name: str,
        model_config: ModelConfig,
        role: RoleBase,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        super().__init__(player_id=player_id, name=name)
        self.role = role
        self.model_config = model_config
        self.client = LLMClient(model_config)
        self.token_tracker = token_tracker or TokenTracker()
        # Optional callback for emitting events (set by runner)
        self._emit_event: Any = None

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the ReAct loop."""
        return (
            "You are an AI agent playing a strategic game of Werewolf.\n"
            f"Your player name is {self.name}. Your role is: {self.role.name}.\n"
            f"\n"
            f"Role: {self.role.description}\n"
            f"\n"
            f"{self.role.prompt_instructions}\n"
            f"\n"
            "=== IMPORTANT RULES ===\n"
            "1. You interact with the game ONLY by calling tools.\n"
            "2. Each response must contain exactly ONE tool call.\n"
            "3. Write: USE: tool_name(arguments)\n"
            "4. Tools marked [TERMINAL] end your turn.\n"
            "5. Do NOT narrate, roleplay, or write fiction. Just reason\n"
            "   briefly about your decision, then call a tool.\n"
            "6. When using speak() or wolf_say(), the argument is your\n"
            "   PUBLIC message. Do NOT leak your role or private strategy\n"
            "   inside speak/wolf_say.\n"
            "7. Keep reasoning SHORT. Decide and act.\n"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_phase(
        self,
        briefing: str,
        toolkit: AgentToolkit,
        *,
        max_rounds: int | None = None,
    ) -> Action:
        """Execute the multi-round ReAct tool loop."""
        rounds = max_rounds or _MAX_ROUNDS
        model_short = self.model_config.model.split(":")[0]
        self._live(f"\033[2m  {self.name}({model_short}) phase start ...\033[0m")
        t0 = _time.monotonic()

        system_prompt = self._build_system_prompt()
        tools_text = toolkit.format_for_prompt()
        known_tool_names = {t.name for t in toolkit.get_terminal_tools()} | {
            t.name for t in toolkit.get_non_terminal_tools()
        }
        terminal_names = [t.name for t in toolkit.get_terminal_tools()]

        # Build the fresh conversation
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{briefing}\n\n{tools_text}"},
        ]

        for round_num in range(rounds):
            is_last_round = round_num == rounds - 1

            # Inject deadline warning on the final round
            if is_last_round:
                messages.append({
                    "role": "user",
                    "content": (
                        f"IMPORTANT: This is your FINAL round. You MUST use a "
                        f"terminal tool now: {', '.join(terminal_names)}. "
                        f"Write USE: {terminal_names[0]}(...) to end your turn."
                    ),
                })

            # Call LLM
            try:
                response = await retry_with_backoff(
                    self.client.chat,
                    messages,
                    temperature=self.model_config.reasoning_temperature,
                )
            except Exception as exc:
                logger.error("LLM call failed for %s (round %d): %s", self.name, round_num, exc)
                self._live(f"\033[91m  {self.name} LLM FAIL (round {round_num}): {exc}\033[0m")
                return NoAction(player_id=self.player_id, reason=f"llm_error:{exc}")

            t_now = _time.monotonic()
            self.token_tracker.record(
                self.player_id,
                response.input_tokens,
                response.output_tokens,
                call_type=f"react_round_{round_num}",
            )

            llm_text = response.content.strip()

            # Print reasoning preview
            clean_text = llm_text.replace("<think>", "").replace("</think>", "").strip()
            preview = clean_text[:300]
            if len(clean_text) > 300:
                preview += "..."
            self._live(
                f"\033[2m  {self.name}({model_short}) round {round_num}: "
                f"{response.input_tokens}in/{response.output_tokens}out "
                f"{t_now - t0:.1f}s\033[0m"
            )
            self._live(f"\033[2m    >> {preview}\033[0m")

            # Emit ReasoningEvent for the LLM's free-text reasoning
            if self._emit_event is not None and round_num == 0:
                self._emit_event(
                    ReasoningEvent(
                        day=0,  # filled by moderator context
                        player_id=self.player_id,
                        reasoning=clean_text,
                        action_type="react",
                    )
                )

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": llm_text})

            # Parse tool call (tries USE: prefix first, then bare tool names)
            tool_name, tool_args = self._parse_tool_call(
                llm_text, known_tools=known_tool_names
            )

            if tool_name is None:
                if is_last_round:
                    break  # will force terminal below
                messages.append({
                    "role": "user",
                    "content": (
                        "You must use a tool. Write USE: tool_name(arguments) "
                        "to proceed. Check the available tools listed above."
                    ),
                })
                continue

            # Invoke the tool
            result = toolkit.invoke(tool_name, tool_args or "")

            if result.is_terminal and result.action is not None:
                t_final = _time.monotonic()
                self._live(
                    f"\033[2m  {self.name}({model_short}) done: "
                    f"{tool_name} -> {result.action.__class__.__name__} "
                    f"total={t_final - t0:.1f}s\033[0m"
                )
                return result.action

            if is_last_round:
                break  # will force terminal below

            # Non-terminal: feed result back as user message
            messages.append({
                "role": "user",
                "content": f"[Tool result: {tool_name}]\n{result.output}",
            })

        # Max rounds exhausted — force a terminal tool
        return self._force_terminal_action(toolkit, model_short, t0)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_tool_prefix(name: str) -> str:
        """Strip common model-added prefixes from tool names.

        Models like gpt-oss sometimes emit ``tool.speak()``,
        ``assistant:speak()``, or ``assistant.vote()``.
        """
        for prefix in _TOOL_NAME_PREFIXES:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    @staticmethod
    def _unwrap_json_tool_call(
        name: str, args: str, known_tools: set[str] | None = None,
    ) -> tuple[str, str]:
        """Unwrap JSON-wrapped tool calls from models like gpt-oss.

        Handles patterns like ``assistant({"name":"get_alive_players","arguments":""})``.
        Returns ``(real_tool_name, real_args)``.
        """
        import json

        # Only unwrap if the outer name is a wrapper (not a real tool)
        if known_tools and name in known_tools:
            return name, args

        # Try to parse the args as JSON with a "name" field
        try:
            data = json.loads(args)
            if isinstance(data, dict) and "name" in data:
                real_name = LLMAgent._strip_tool_prefix(str(data["name"]))
                real_args = str(data.get("arguments", ""))
                return real_name, real_args
        except (json.JSONDecodeError, TypeError):
            pass

        return name, args

    @staticmethod
    def _parse_tool_call(
        text: str,
        known_tools: set[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Parse ``USE: tool_name(args)`` from LLM output.

        Falls back to matching bare ``tool_name(args)`` if a known tool
        name is recognized.  Returns ``(tool_name, args_str)`` or
        ``(None, None)`` if no tool call is found.
        """
        # Primary: USE: tool_name(args)
        match = _TOOL_CALL_RE.search(text)
        if match:
            tool_name = LLMAgent._strip_tool_prefix(match.group(1))
            args = match.group(2)
            if args is not None:
                args = args.strip()
            # Unwrap JSON-wrapped tool calls: assistant({"name":"tool","arguments":""})
            tool_name, args = LLMAgent._unwrap_json_tool_call(
                tool_name, args or "", known_tools,
            )
            return tool_name, args

        # Fallback: bare tool_name(args) for known tool names
        if known_tools:
            for m in _BARE_TOOL_RE.finditer(text):
                raw_name = LLMAgent._strip_tool_prefix(m.group(1))
                raw_args = m.group(2).strip() if m.group(2) else ""
                # Unwrap JSON wrapper
                raw_name, raw_args = LLMAgent._unwrap_json_tool_call(
                    raw_name, raw_args, known_tools,
                )
                if raw_name in known_tools:
                    return raw_name, raw_args

        return None, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _force_terminal_action(
        self,
        toolkit: AgentToolkit,
        model_short: str,
        t0: float,
    ) -> Action:
        """Force a terminal tool when the agent exhausted its rounds."""
        terminal_tools = toolkit.get_terminal_tools()

        # Prefer pass_turn (always safe)
        for t in terminal_tools:
            if t.name == "pass_turn":
                result = toolkit.invoke("pass_turn", "")
                if result.is_terminal and result.action is not None:
                    t_final = _time.monotonic()
                    self._live(
                        f"\033[93m  {self.name}({model_short}) forced "
                        f"pass_turn total={t_final - t0:.1f}s\033[0m"
                    )
                    return result.action

        # Shouldn't happen, but fall back to NoAction
        self._live(
            f"\033[93m  {self.name} max rounds reached, "
            f"returning NoAction\033[0m"
        )
        return NoAction(player_id=self.player_id, reason="max_rounds_forced")

    @staticmethod
    def _live(msg: str) -> None:
        """Print a live status message to stdout."""
        print(msg, flush=True)
