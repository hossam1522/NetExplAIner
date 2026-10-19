import logging
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def configure_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "tests.log"
        logger = logging.getLogger("netexplainer")
        logger.handlers.clear()

        formatter = logging.Formatter("%(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        yield


class FakeChatModel:
    """Minimal stand-in for a LangChain chat model.

    ``responder(messages) -> AIMessage | str`` decides each reply; the prompts
    received are stored in ``calls``.
    """

    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls.append(list(messages))
        reply = self.responder(messages)
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)


@pytest.fixture
def fake_chat_model():
    return FakeChatModel
