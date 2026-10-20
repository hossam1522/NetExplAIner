"""Thin wrapper around a LangChain chat model: tool-calling loop + usage stats."""

import logging
import time
from dataclasses import asdict, dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage

from netexplainer.llm.tools import TOOLS

logger = logging.getLogger(__name__)

# Tool modes: ``none`` (no tools), ``callable`` (tools bound and executed) and
# ``declared`` (tools bound/described but never executed: the prompt effect).
TOOL_MODES = ("none", "callable", "declared")
TOOL_UNAVAILABLE = "Error: the tool is not available. Answer without using tools."
MAX_TOOL_ROUNDS = 10


@dataclass
class Usage:
    """Counters accumulated over the LLM calls of one pipeline run."""

    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    tool_calls_blocked: int = 0  # calls answered with TOOL_UNAVAILABLE
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class LLMClient:
    def __init__(self, chat_model: BaseChatModel, tools: bool | str = False):
        """
        Args:
            chat_model: LangChain chat model (see ``backends.build_chat_model``).
            tools: tool mode, one of :data:`TOOL_MODES` (``True`` = callable,
                ``False`` = none). ``declared`` binds the calculator so the
                model sees its definition, but a call is answered with a
                constant error message (and counted in ``tool_calls_blocked``).
        """
        mode = {True: "callable", False: "none"}.get(tools, tools)
        if mode not in TOOL_MODES:
            raise ValueError(f"Unknown tool mode {tools!r}, expected {TOOL_MODES}")
        self.llm = chat_model
        self.tool_mode = mode
        self.tools = mode != "none"  # tools are bound to the model
        self._tools_by_name = {t.name: t for t in TOOLS}
        self.llm_with_tools = chat_model.bind_tools(tools=TOOLS) if self.tools else None
        self.usage = Usage()

    def reset_usage(self) -> Usage:
        """Return the accumulated usage and start counting from zero."""
        usage, self.usage = self.usage, Usage()
        return usage

    def run_tool(self, name: str, expression: str) -> str:
        """Execute tool ``name`` (counted); in ``declared`` mode it only fails."""
        self.usage.tool_calls += 1
        if self.tool_mode != "callable" or name not in self._tools_by_name:
            self.usage.tool_calls_blocked += 1
            return TOOL_UNAVAILABLE
        return self._tools_by_name[name].invoke(expression)

    def _invoke_once(self, messages: list[BaseMessage], plain: bool = False):
        runnable = self.llm_with_tools if self.tools and not plain else self.llm
        start = time.perf_counter()
        response = runnable.invoke(messages)
        self.usage.latency_s += time.perf_counter() - start
        self.usage.llm_calls += 1
        meta = getattr(response, "usage_metadata", None) or {}
        self.usage.tokens_in += meta.get("input_tokens", 0) or 0
        self.usage.tokens_out += meta.get("output_tokens", 0) or 0
        return response

    def invoke_plain(self, messages: list[BaseMessage]) -> str:
        """One call without native tool binding (used by the text-based ReAct)."""
        return self._invoke_once(messages, plain=True).content

    def invoke(self, messages: list[BaseMessage]) -> str:
        """Call the model; while it requests tools, run them and call it again.

        Returns the content of the first response without tool calls (or of
        the last one after ``MAX_TOOL_ROUNDS`` rounds of tool calls).
        """
        response = self._invoke_once(messages)
        rounds = 0
        while response.tool_calls and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            tool_responses = []
            for tool_call in response.tool_calls:
                result = self.run_tool(
                    tool_call["name"], tool_call["args"].get("expression", "")
                )
                tool_responses.append(
                    ToolMessage(
                        content=result,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            messages.append(response)
            messages.extend(tool_responses)
            response = self._invoke_once(messages)
        return response.content
