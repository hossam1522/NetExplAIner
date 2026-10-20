"""Per-iteration result records (JSONL) and parser for the legacy logs.

One JSON object per (model, strategy, tools, trace, question). See README
("Results format") for the field list.
"""

import json
import re
import subprocess
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from netexplainer import __version__
from netexplainer.evaluation.judge import parse_similarity, parse_verdict
from netexplainer.llm.pipeline import DEFAULT_MIXED_DIRECT, uses_decomposition

SCHEMA_VERSION = 1


def tool_mode_of(record: dict) -> str:
    """``none``/``callable``/``declared``; derived from ``tools`` in old records."""
    return record.get("tool_mode") or ("callable" if record.get("tools") else "none")


def condition_label(strategy: str, tools: bool | str) -> str:
    """Experimental condition, e.g. ``decomp+tools`` (calculator callable) or
    ``decomp+declared`` (calculator declared but not callable)."""
    mode = {True: "callable", False: "none"}.get(tools, tools)
    suffix = {"callable": "+tools", "declared": "+declared"}.get(mode, "")
    return strategy + suffix


def record_key(record: dict) -> tuple:
    """Identity of a record, used for resuming and for paired tests."""
    return (
        record["model"],
        record["strategy"],
        tool_mode_of(record),
        record["file"],
        record["question"],
    )


@lru_cache(maxsize=1)
def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def make_record(**fields) -> dict:
    """Fill the derived/common fields of a result record."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "netexplainer_version": __version__,
        "git_commit": git_commit(),
    }
    record.update(fields)
    record.setdefault("tool_mode", tool_mode_of(record))
    record["tools"] = record["tool_mode"] == "callable"
    record["condition"] = condition_label(record["strategy"], record["tool_mode"])
    record["verdict"] = parse_verdict(record.get("judge_raw"))
    record["correct"] = {"YES": True, "NO": False}.get(record["verdict"])
    record["subq_similarity"] = parse_similarity(record.get("subq_similarity_raw"))
    return record


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def completed_keys(path: Path) -> set[tuple]:
    """Keys already present in ``path`` (empty if it does not exist)."""
    if not Path(path).exists():
        return set()
    return {record_key(r) for r in read_jsonl(path)}


# Line written by the original evaluator (and still by the runner), e.g.
# "... - INFO - Model: phi4_tools, File: a.pcap, Question: Q?, Subquestions
#  Eval: 80, Answer Eval: YES"
_LEGACY_LINE = re.compile(
    r"Model: (?P<model>.+?), File: (?P<file>.+?), Question: (?P<question>.+?), "
    r"Subquestions Eval: (?P<subq>.*?), Answer Eval: (?P<answer>.*)$"
)


def parse_legacy_log(
    lines: Iterable[str],
    strategy: str = "decomp",
    questions: list[str] | None = None,
    direct_questions: Iterable[int] = DEFAULT_MIXED_DIRECT,
) -> Iterator[dict]:
    """Convert the original ``netexplainer.log`` result lines to records.

    The old logs do not say which strategy was used, so it must be given
    (``decomp`` = full decomposition, ``mixed``). ``questions`` (ordered as in
    questions.yaml) is used to recover the question index. A ``_tools`` suffix
    in the model name means tools were enabled. Fields that were not logged
    (answers, sub-questions, usage...) are ``None``.
    """
    index = {q: i for i, q in enumerate(questions or [], start=1)}
    for line in lines:
        match = _LEGACY_LINE.search(line.rstrip("\n"))
        if not match:
            continue
        model = match["model"]
        tools = model.endswith("_tools")
        if tools:
            model = model[: -len("_tools")]
        q_index = index.get(match["question"])
        decomposed = (
            uses_decomposition(strategy, q_index, direct_questions)
            if q_index is not None
            else None
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "source": "legacy_log",
            "model": model,
            "strategy": strategy,
            "tools": tools,
            "condition": condition_label(strategy, tools),
            "file": match["file"],
            "question_index": q_index,
            "question": match["question"],
            "decomposed": decomposed,
            "judge_raw": match["answer"],
            "verdict": parse_verdict(match["answer"]),
            "subq_similarity_raw": match["subq"],
            "subq_similarity": (
                parse_similarity(match["subq"]) if decomposed is not False else None
            ),
        }
        record["correct"] = {"YES": True, "NO": False}.get(record["verdict"])
        yield record


def load_records(
    paths: Iterable[Path], legacy_strategy: str = "decomp", questions=None
) -> list[dict]:
    """Load ``.jsonl`` result files and/or legacy ``.log`` files."""
    records = []
    for path in map(Path, paths):
        if path.suffix == ".jsonl":
            records.extend(read_jsonl(path))
        else:
            with open(path, encoding="utf-8") as f:
                records.extend(parse_legacy_log(f, legacy_strategy, questions))
    return records
