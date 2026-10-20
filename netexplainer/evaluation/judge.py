"""LLM-as-a-judge: answer correctness (YES/NO) and sub-question similarity."""

import logging
import re
import time
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from netexplainer.llm import prompts

logger = logging.getLogger(__name__)

# Judge used in the paper: Gemma 3 27B through the Google API, temperature 0.
DEFAULT_JUDGE_BACKEND = "google"
DEFAULT_JUDGE_MODEL = "gemma-3-27b-it"
DEFAULT_JUDGE_TEMPERATURE = 0.0

_VERDICT = re.compile(r"^\W*(YES|NO)\b", re.IGNORECASE)


def parse_verdict(raw) -> str:
    """Normalize the judge output to ``YES``, ``NO`` or ``PROBLEM``."""
    if raw is None:
        return "PROBLEM"
    match = _VERDICT.match(str(raw))
    return match.group(1).upper() if match else "PROBLEM"


def parse_similarity(raw) -> float | None:
    """Parse the judge similarity (e.g. ``"80%"``) as a float in [0, 100]."""
    if raw is None:
        return None
    try:
        return float(str(raw).replace("%", ""))
    except ValueError:
        return None


class Judge:
    def __init__(
        self,
        chat_model: BaseChatModel,
        name: str = DEFAULT_JUDGE_MODEL,
        delay: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """
        Args:
            chat_model: model used as judge.
            name: judge model name (stored in the results).
            delay: seconds to wait before each judge call (API rate limits).
        """
        self.llm = chat_model
        self.name = name
        self.delay = delay
        self._sleep = sleep

    def _ask(self, template: str, **values) -> str:
        if self.delay:
            self._sleep(self.delay)
        messages = ChatPromptTemplate.from_template(template).format_messages(**values)
        return self.llm.invoke(messages).content

    def evaluate_subquestions(
        self, question: str, subquestions: list, reference: list
    ) -> str:
        """Similarity (0-100, raw judge text) between generated and reference
        sub-questions. Lists are rendered with ``str()`` as in the paper."""
        answer = self._ask(
            prompts.JUDGE_SUBQUESTIONS,
            question=question,
            subquestions_LLM=subquestions,
            subquestions=reference,
        )
        logger.debug(
            f"Question: {question}, Subquestions LLM: {subquestions}, Subquestions: {reference}, Similarity: {answer}"
        )
        return answer

    def evaluate_answer(self, question: str, answer_llm: str, ground_truth) -> str:
        """Raw judge verdict (expected ``YES``/``NO``) for ``answer_llm``."""
        answer = self._ask(
            prompts.JUDGE_ANSWER,
            question=question,
            answer_LLM=answer_llm,
            answer=ground_truth,
        )
        logger.debug(
            f"Question: {question}, Answer LLM: {answer_llm}, Answer: {ground_truth}, Comparison: {answer}"
        )
        return answer


class NullJudge:
    """Judge that does not judge (``--judge-backend none``): raw outputs are
    ``None``, so records are stored as ``PROBLEM`` until they are re-judged
    with ``netexplainer rejudge``."""

    name = "none"

    def evaluate_subquestions(self, question, subquestions, reference):
        return None

    def evaluate_answer(self, question, answer_llm, ground_truth):
        return None
