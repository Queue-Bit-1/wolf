"""Agent toolkit: tool definitions and invocation for the tool-based agent architecture."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDefinition:
    """A single tool available to an agent.

    Parameters
    ----------
    name:
        Short identifier used in ``USE: name(args)`` syntax.
    description:
        Human-readable description shown to the LLM.
    parameters:
        Human-readable parameter description (e.g. ``"player_name"``).
        Empty string for no-arg tools.
    is_terminal:
        If True, invoking this tool ends the agent's turn.
    handler:
        Callable ``(args_str) -> str | Action``.
        Non-terminal tools return a plain-text string.
        Terminal tools return an ``Action`` object.
    """

    name: str
    description: str
    parameters: str
    is_terminal: bool
    handler: Callable[..., Any]


@dataclass
class ToolResult:
    """Result of invoking a tool."""

    tool_name: str
    output: str
    is_terminal: bool
    action: Any = None  # Set only when a terminal tool returns an Action


class AgentToolkit:
    """Collection of tools available to an agent for a specific phase.

    The toolkit is built by :class:`ToolFactory` and passed to
    ``agent.run_phase(briefing, toolkit)``.  Tools are invoked by name
    and always return a :class:`ToolResult`.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        return dict(self._tools)

    def get_terminal_tools(self) -> list[ToolDefinition]:
        """Return all terminal tools."""
        return [t for t in self._tools.values() if t.is_terminal]

    def get_non_terminal_tools(self) -> list[ToolDefinition]:
        """Return all non-terminal (information) tools."""
        return [t for t in self._tools.values() if not t.is_terminal]

    def invoke(self, tool_name: str, args: str = "") -> ToolResult:
        """Invoke a tool by name and return the result.

        Parameters
        ----------
        tool_name:
            The registered tool name.
        args:
            Raw argument string from the LLM.

        Returns
        -------
        ToolResult
            Contains the output text (and an Action if terminal).
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys()))
            return ToolResult(
                tool_name=tool_name,
                output=f"Error: unknown tool '{tool_name}'. Available tools: {available}",
                is_terminal=False,
            )

        try:
            result = tool.handler(args.strip()) if args.strip() else tool.handler("")
        except Exception as exc:
            logger.exception("Tool %s raised an exception", tool_name)
            return ToolResult(
                tool_name=tool_name,
                output=f"Error invoking {tool_name}: {exc}",
                is_terminal=False,
            )

        if tool.is_terminal:
            # Terminal tools return Action objects
            from wolf.engine.actions import Action
            if isinstance(result, Action):
                return ToolResult(
                    tool_name=tool_name,
                    output=f"[Action submitted: {tool_name}]",
                    is_terminal=True,
                    action=result,
                )
            # If the handler returned a string, treat it as an error
            return ToolResult(
                tool_name=tool_name,
                output=f"Error: terminal tool {tool_name} did not return an Action: {result}",
                is_terminal=False,
            )

        # Non-terminal tools return plain text
        if isinstance(result, str):
            return ToolResult(
                tool_name=tool_name,
                output=result,
                is_terminal=False,
            )

        return ToolResult(
            tool_name=tool_name,
            output=str(result),
            is_terminal=False,
        )

    def format_for_prompt(self) -> str:
        """Format all registered tools as a text block for the LLM prompt.

        Returns a human-readable listing of tools the agent can invoke.
        """
        lines = ["=== Available Tools ==="]
        lines.append("To use a tool, write: USE: tool_name(arguments)")
        lines.append("Tools marked [TERMINAL] end your turn.\n")

        for tool in self._tools.values():
            terminal_tag = " [TERMINAL]" if tool.is_terminal else ""
            if tool.parameters:
                sig = f"{tool.name}({tool.parameters})"
            else:
                sig = f"{tool.name}()"
            lines.append(f"  {sig}{terminal_tag}")
            lines.append(f"    {tool.description}")
            lines.append("")

        return "\n".join(lines)
