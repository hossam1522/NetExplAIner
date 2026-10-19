"""Experiment loop: models x conditions x traces x questions -> JSONL records."""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from netexplainer.dataset import Dataset
from netexplainer.evaluation.judge import Judge
from netexplainer.evaluation.records import (
    append_jsonl,
    completed_keys,
    make_record,
    record_key,
)
from netexplainer.llm import (
    DEFAULT_MIXED_DIRECT,
    GenerationParams,
    LLMClient,
    Pipeline,
    build_chat_model,
    resolve_model,
)

logger = logging.getLogger(__name__)

CAPTURE_EXTENSIONS = (".pcap", ".pcapng", ".cap")


@dataclass
class RunConfig:
    models: list[str]
    strategies: list[str]
    data_dir: Path
    questions_path: Path
    output: Path
    tools: list[bool] = field(default_factory=lambda: [False])
    backend: str = "ollama"  # for models that are not in the registry
    base_url: str | None = None
    api_key: str | None = None
    params: GenerationParams = GenerationParams()  # overrides registry values
    max_files: int | None = None
    max_attempts: int = 10
    direct_questions: tuple[int, ...] = DEFAULT_MIXED_DIRECT
    resume: bool = False
    request_delay: float | None = None  # None: 2.5 s for Google, 0 otherwise
    error_delay: float = 10.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def conditions(strategies: list[str], tools: list[bool]) -> list[tuple[str, bool]]:
    """(strategy, tools) pairs to run. ``decomp_tools`` means decomp + tools."""
    out = []
    for strategy in strategies:
        pairs = (
            [("decomp", True)]
            if strategy == "decomp_tools"
            else [(strategy, t) for t in tools]
        )
        out += [p for p in pairs if p not in out]
    return out


def list_traces(data_dir: Path, max_files: int | None = None) -> list[Path]:
    """Capture files in ``data_dir`` in a deterministic (sorted) order."""
    files = sorted(
        p for p in Path(data_dir).iterdir() if p.suffix.lower() in CAPTURE_EXTENSIONS
    )
    return files[:max_files] if max_files is not None else files


def _evaluate_question(
    pipeline, client, judge, dataset, cfg, strategy, idx, question, sleep
):
    """Run one question with retries; return the record fields.

    Same retry policy as in the paper: the whole pipeline + judge is retried
    (up to ``cfg.max_attempts``) until it runs and both judge calls succeed.
    Usage counters refer to the last attempt.
    """
    errors: list[str] = []
    result, subq_raw, judge_raw = None, None, None
    for attempt in range(1, cfg.max_attempts + 1):
        client.reset_usage()
        result, subq_raw, judge_raw = None, None, None
        try:
            result = pipeline.run(strategy, question, idx, cfg.direct_questions)
        except NotImplementedError:
            raise
        except Exception as e:
            logger.error(
                f"Error processing question {question} in file {Path(dataset.processed_file).name}: {e}"
            )
            errors.append(f"attempt {attempt}: pipeline: {e}")
            sleep(cfg.error_delay)
            continue
        ok = True
        if result.decomposed:
            try:
                subq_raw = judge.evaluate_subquestions(
                    question,
                    result.subquestions,
                    dataset.questions_subquestions[question],
                )
            except Exception as e:
                logger.error(f"Error evaluating subquestions: {e}")
                errors.append(f"attempt {attempt}: judge subquestions: {e}")
                ok = False
        try:
            judge_raw = judge.evaluate_answer(
                question, result.final_answer, dataset.questions_answers[question]
            )
        except Exception as e:
            logger.error(f"Error evaluating answers: {e}")
            errors.append(f"attempt {attempt}: judge answer: {e}")
            ok = False
        if ok:
            break
    usage = client.reset_usage()
    return {
        "decomposed": result.decomposed if result else None,
        "subquestions": result.subquestions if result else None,
        "intermediate_answers": result.intermediate_answers if result else None,
        "final_answer": result.final_answer if result else None,
        "reference_subquestions": dataset.questions_subquestions[question],
        "ground_truth": str(dataset.questions_answers.get(question)),
        "judge_model": judge.name,
        "judge_raw": judge_raw,
        "subq_similarity_raw": subq_raw,
        "attempts": attempt,
        **usage.to_dict(),
        "errors": errors,
    }


def run_experiment(
    cfg: RunConfig,
    judge: Judge,
    chat_model_factory: Callable = build_chat_model,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run every configuration in ``cfg`` and append records to ``cfg.output``.

    Returns the number of records written.
    """
    done = completed_keys(cfg.output) if cfg.resume else set()
    if not cfg.resume and Path(cfg.output).exists() and Path(cfg.output).stat().st_size:
        raise FileExistsError(
            f"{cfg.output} already exists; use --resume or another --output-dir"
        )
    traces = list_traces(cfg.data_dir, cfg.max_files)
    written = 0
    for model in cfg.models:
        spec = resolve_model(model, cfg.backend)
        params = spec.params.override(cfg.params)
        chat_model = chat_model_factory(
            spec.backend,
            spec.name,
            params,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            extra=spec.extra,
        )
        delay = cfg.request_delay
        if delay is None:
            delay = 2.5 if spec.backend == "google" else 0.0
        clients = {
            t: LLMClient(chat_model, tools=t)
            for _, t in conditions(cfg.strategies, cfg.tools)
        }
        for trace in traces:
            try:
                logger.debug(f"Processing file: {trace.name} with model: {model}")
                dataset = Dataset(str(trace), str(cfg.questions_path), spec.context)
                text = Path(dataset.processed_file).read_text()
            except Exception as e:
                logger.error(
                    f"Error processing file {trace.name} with model {model}: {e}"
                )
                sleep(cfg.error_delay)
                continue
            for strategy, tools in conditions(cfg.strategies, cfg.tools):
                client = clients[tools]
                pipeline = Pipeline(client, text, spec.name, delay, sleep)
                for idx, question in enumerate(dataset.questions_subquestions, start=1):
                    base = {
                        "model": model,
                        "strategy": strategy,
                        "tools": tools,
                        "file": trace.name,
                        "question": question,
                    }
                    if record_key(base) in done:
                        continue
                    logger.debug(f"Processing question: {question} with model: {model}")
                    fields = _evaluate_question(
                        pipeline,
                        client,
                        judge,
                        dataset,
                        cfg,
                        strategy,
                        idx,
                        question,
                        sleep,
                    )
                    record = make_record(
                        run_id=cfg.run_id,
                        model_name=spec.name,
                        backend=spec.backend,
                        question_index=idx,
                        seed=params.seed,
                        generation_params=params.to_dict(),
                        **base,
                        **fields,
                    )
                    append_jsonl(cfg.output, record)
                    written += 1
                    sim = record["subq_similarity_raw"] if record["decomposed"] else 100
                    logger.info(
                        f"Model: {model}{'_tools' if tools else ''}, File: {trace.name}, "
                        f"Question: {question}, Subquestions Eval: {sim}, "
                        f"Answer Eval: {record['verdict']}"
                    )
    return written
