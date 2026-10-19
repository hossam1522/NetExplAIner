"""Question-answering strategies over a network trace.

Decomposition pipeline (``decomp``)::

    q --decompose--> {q_1..q_k} --answer(trace)--> {a_1..a_k} --g--> a

where ``g`` (:meth:`Pipeline.aggregate`, prompt ``AGGREGATION``) synthesizes
the final answer from the question and the (q_i, a_i) pairs only.
"""

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from netexplainer.llm import prompts
from netexplainer.llm.client import LLMClient

logger = logging.getLogger(__name__)

STRATEGIES = ("direct", "cot", "decomp", "decomp_tools", "mixed", "react")
# Questions answered directly by the ``mixed`` strategy (1-based indexes in
# questions.yaml): the simple retrieval questions Q1, Q2 and Q4 of the paper.
DEFAULT_MIXED_DIRECT = (1, 2, 4)


@dataclass
class PipelineResult:
    final_answer: str
    decomposed: bool = False
    subquestions: list[str] = field(default_factory=list)
    intermediate_answers: list[str] = field(default_factory=list)


def uses_decomposition(
    strategy: str,
    question_index: int,
    direct_questions: Iterable[int] = DEFAULT_MIXED_DIRECT,
) -> bool:
    """Whether ``strategy`` decomposes the question at 1-based ``question_index``."""
    if strategy == "mixed":
        return question_index not in set(direct_questions)
    return strategy in ("decomp", "decomp_tools")


class Pipeline:
    def __init__(
        self,
        client: LLMClient,
        trace: str,
        model_label: str = "",
        request_delay: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """
        Args:
            client: LLM client (tools are configured there).
            trace: the network trace rendered as text.
            model_label: used only for logging.
            request_delay: seconds to wait before each answering/aggregation
                call (rate limiting of remote APIs; 2.5 s for Google in the
                paper, 0 for local backends).
        """
        self.client = client
        self.trace = trace
        self.model = model_label
        self.request_delay = request_delay
        self._sleep = sleep

    def _call(self, template: str, **values) -> str:
        messages = ChatPromptTemplate.from_template(template).format_messages(**values)
        return self.client.invoke(messages)

    def _pause(self) -> None:
        if self.request_delay:
            self._sleep(self.request_delay)

    def decompose(self, question: str) -> list[str]:
        """Generate the sub-questions of ``question`` (one per non-empty line)."""
        text = self._call(prompts.DECOMPOSITION, question=question)
        subquestions = [q.strip() for q in text.split("\n") if q.strip()]
        logger.debug(
            f"Model: {self.model}, Question: {question}, Sub-questions generated: {subquestions}"
        )
        return subquestions

    def answer(self, question: str, template: str = prompts.ANSWER) -> str:
        """Answer ``question`` looking at the trace."""
        self._pause()
        answer = self._call(template, traces=self.trace, question=question)
        logger.debug(f"Model: {self.model}, Question: {question}, Answer: {answer}")
        return answer

    def aggregate(self, question: str, subquestions: list, answers: list) -> str:
        """Aggregation function g(q, {(q_i, a_i)}) -> final answer."""
        self._pause()
        final_answer = self._call(
            prompts.AGGREGATION,
            context=prompts.format_qa_pairs(subquestions, answers),
            question=question,
        )
        logger.debug(
            f"Model: {self.model}, Question: {question}, Final answer: {final_answer}"
        )
        return final_answer

    def run_decomposed(self, question: str) -> PipelineResult:
        subquestions = self.decompose(question)
        answers = [self.answer(sq) for sq in subquestions]
        final = self.aggregate(question, subquestions, answers)
        return PipelineResult(final, True, subquestions, answers)

    def run(
        self,
        strategy: str,
        question: str,
        question_index: int,
        direct_questions: Iterable[int] = DEFAULT_MIXED_DIRECT,
    ) -> PipelineResult:
        """Answer ``question`` (1-based ``question_index``) with ``strategy``.

        ``decomp_tools`` behaves as ``decomp``: tools are enabled in the client.
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}, expected {STRATEGIES}")
        if strategy == "react":
            raise NotImplementedError("The 'react' strategy is not implemented yet")
        if uses_decomposition(strategy, question_index, direct_questions):
            return self.run_decomposed(question)
        if strategy == "cot":
            return PipelineResult(self.answer(question, template=prompts.COT))
        return PipelineResult(self.answer(question))
