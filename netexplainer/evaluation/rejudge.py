"""Judge (or re-judge) the answers of an existing results file.

Useful to evaluate runs made with ``--judge-backend none`` (e.g. while the GPUs
are busy with the evaluated model) or to compare judges on the same answers.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from netexplainer.evaluation.judge import Judge
from netexplainer.evaluation.records import (
    append_jsonl,
    completed_keys,
    make_record,
    read_jsonl,
    record_key,
)

logger = logging.getLogger(__name__)

_DERIVED = ("condition", "verdict", "correct", "subq_similarity")


def rejudge(
    source: Path,
    output: Path,
    judge: Judge,
    max_attempts: int = 10,
    error_delay: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Write to ``output`` every record of ``source`` judged by ``judge``.

    Records already in ``output`` are skipped (resumable). Records without a
    final answer are copied as they are. Returns the number of records written.
    """
    done = completed_keys(output)
    written = 0
    for record in read_jsonl(source):
        if record_key(record) in done:
            continue
        new = {k: v for k, v in record.items() if k not in _DERIVED}
        if record.get("final_answer") is not None:
            for attempt in range(1, max_attempts + 1):
                try:
                    subq = None
                    if record.get("decomposed"):
                        subq = judge.evaluate_subquestions(
                            record["question"],
                            record["subquestions"],
                            record["reference_subquestions"],
                        )
                    raw = judge.evaluate_answer(
                        record["question"],
                        record["final_answer"],
                        record["ground_truth"],
                    )
                    new.update(subq_similarity_raw=subq, judge_raw=raw)
                    break
                except Exception as e:
                    logger.error(f"Judge error (attempt {attempt}): {e}")
                    sleep(error_delay)
            new.update(judge_model=judge.name)
        append_jsonl(output, make_record(**new))
        written += 1
    return written
