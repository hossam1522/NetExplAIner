"""Arguments of ``run`` / ``rejudge`` (models, strategy, judge, data)."""

import argparse
from pathlib import Path

from netexplainer.dataset import QUESTION_SETS
from netexplainer.evaluation.judge import (
    DEFAULT_JUDGE_BACKEND,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
)
from netexplainer.llm import BACKENDS, STRATEGIES, GenerationParams

DATA_DIR = Path("netexplainer/data")
DEFAULT_QUESTIONS = Path(__file__).parent / "data" / "questions.yaml"


def int_list(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def add_judge_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("judge")
    g.add_argument(
        "--judge-backend",
        choices=(*BACKENDS, "none"),
        default=DEFAULT_JUDGE_BACKEND,
        help="'none' skips judging (use 'rejudge' later)",
    )
    g.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    g.add_argument("--judge-base-url")
    g.add_argument("--judge-temperature", type=float, default=DEFAULT_JUDGE_TEMPERATURE)
    g.add_argument(
        "--judge-num-ctx",
        type=int,
        help="Ollama judge context window (long CoT/ReAct answers need more "
        "than the Ollama default)",
    )
    g.add_argument(
        "--judge-delay",
        type=float,
        default=0.5,
        help="seconds between judge calls (API rate limits)",
    )


def build_judge(args):
    from netexplainer.evaluation.judge import Judge, NullJudge
    from netexplainer.llm import build_chat_model

    if args.judge_backend == "none":
        return NullJudge()
    return Judge(
        build_chat_model(
            args.judge_backend,
            args.judge_model,
            GenerationParams(
                temperature=args.judge_temperature, num_ctx=args.judge_num_ctx
            ),
            base_url=args.judge_base_url,
        ),
        name=args.judge_model,
        delay=args.judge_delay,
    )


def add_run_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("models")
    g.add_argument(
        "--models",
        nargs="+",
        help="model aliases (see llm/models.py) or raw "
        "backend model names; default: 'models' in the questions file",
    )
    g.add_argument(
        "--backend",
        choices=BACKENDS,
        default="ollama",
        help="backend for models not in the registry (default: ollama)",
    )
    g.add_argument(
        "--base-url",
        help="server URL (Ollama or OpenAI-compatible, e.g. "
        "http://localhost:8000/v1 for vLLM)",
    )
    g = p.add_argument_group("strategy")
    g.add_argument(
        "--strategy",
        nargs="+",
        choices=STRATEGIES,
        default=["decomp"],
        help="one or more strategies (default: decomp)",
    )
    g.add_argument(
        "--direct-questions",
        type=int_list,
        default=(1, 2, 4),
        metavar="I,J,..",
        help="1-based questions answered directly by 'mixed' (default: 1,2,4)",
    )
    tools = g.add_mutually_exclusive_group()
    tools.add_argument("--tools", action="store_true", help="bind the calculator tool")
    tools.add_argument(
        "--tools-declared-not-callable",
        action="store_true",
        help="calculator declared in the prompt/system but never executed (a "
        "call gets a constant error message and is counted): prompt-effect ablation",
    )
    tools.add_argument(
        "--tool-ablation",
        action="store_true",
        help="run every strategy both without and with tools",
    )
    g.add_argument(
        "--react-max-steps",
        type=int,
        default=8,
        help="maximum model calls of the 'react' loop (default: 8)",
    )
    g = p.add_argument_group("generation (unset = backend default, as in the paper)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--seed", type=int)
    g.add_argument(
        "--num-ctx", type=int, help="context window (overrides the registry)"
    )
    g.add_argument("--max-tokens", type=int)
    add_judge_arguments(p)
    g = p.add_argument_group("data and execution")
    g.add_argument("--data-dir", type=Path, default=DATA_DIR / "cleaned")
    g.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    g.add_argument(
        "--question-set",
        choices=QUESTION_SETS,
        default="core",
        help="core: the 9 questions of the paper; extended: Q10-Q15; all",
    )
    g.add_argument("--max-files", type=int, help="use only the first N traces (sorted)")
    g.add_argument(
        "--max-packets",
        type=int,
        metavar="N",
        help="only traces with at most N packets (as clean-data N), "
        "e.g. 256/512 for the scale study (raise --num-ctx accordingly)",
    )
    g.add_argument(
        "--min-packets",
        type=int,
        metavar="N",
        help="only traces with at least N packets (to study a size band)",
    )
    g.add_argument("--output-dir", type=Path, default=Path("results"))
    g.add_argument(
        "--resume",
        action="store_true",
        help="skip items already in <output-dir>/results.jsonl",
    )
    g.add_argument("--max-attempts", type=int, default=10)
    g.add_argument(
        "--request-delay",
        type=float,
        help="seconds between LLM calls (default: 2.5 for Google, else 0)",
    )
