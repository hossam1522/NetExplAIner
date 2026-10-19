"""Thin wrapper around a LangChain chat model: tool-calling loop + usage stats."""

import logging
import time
from dataclasses import asdict, dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage

from netexplainer.llm.tools import TOOLS

logger = logging.getLogger(__name__)


@dataclass
class Usage:
    """Counters accumulated over the LLM calls of one pipeline run."""

    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class LLMClient:
    def __init__(self, chat_model: BaseChatModel, tools: bool = False):
        """
        Args:
            chat_model: LangChain chat model (see ``backends.build_chat_model``).
            tools: bind the external tools (calculator) to every call.
        """
        self.llm = chat_model
        self.tools = tools
        self._tools_by_name = {t.name: t for t in TOOLS}
        self.llm_with_tools = chat_model.bind_tools(tools=TOOLS) if tools else None
        self.usage = Usage()

    def reset_usage(self) -> Usage:
        """Return the accumulated usage and start counting from zero."""
        usage, self.usage = self.usage, Usage()
        return usage

    def _invoke_once(self, messages: list[BaseMessage]):
        runnable = self.llm_with_tools if self.tools else self.llm
        start = time.perf_counter()
        response = runnable.invoke(messages)
        self.usage.latency_s += time.perf_counter() - start
        self.usage.llm_calls += 1
        meta = getattr(response, "usage_metadata", None) or {}
        self.usage.tokens_in += meta.get("input_tokens", 0) or 0
        self.usage.tokens_out += meta.get("output_tokens", 0) or 0
        return response

    def invoke(self, messages: list[BaseMessage]) -> str:
        """Call the model; while it requests tools, run them and call it again.

        Returns the content of the first response without tool calls.
        """
        response = self._invoke_once(messages)
        while response.tool_calls:
            tool_responses = []
            for tool_call in response.tool_calls:
                self.usage.tool_calls += 1
                if tool_call["name"] in self._tools_by_name:
                    tool = self._tools_by_name[tool_call["name"]]
                    result = tool.invoke(tool_call["args"]["expression"])
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
