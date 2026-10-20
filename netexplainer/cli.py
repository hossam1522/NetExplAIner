"""Command line interface: ``python -m netexplainer <command> ...``."""

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from netexplainer import __version__
from netexplainer.evaluation.judge import (
    DEFAULT_JUDGE_BACKEND,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
)
from netexplainer.llm import BACKENDS, STRATEGIES, GenerationParams

DATA_DIR = Path("netexplainer/data")
DEFAULT_QUESTIONS = Path(__file__).parent / "data" / "questions.yaml"
DEFAULT_LOG = DATA_DIR / "evaluation" / "netexplainer.log"

logger = logging.getLogger("netexplainer")


def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _add_judge_arguments(p: argparse.ArgumentParser) -> None:
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
        "--judge-delay",
        type=float,
        default=0.5,
        help="seconds between judge calls (API rate limits)",
    )


def _build_judge(args):
    from netexplainer.evaluation.judge import Judge, NullJudge
    from netexplainer.llm import build_chat_model

    if args.judge_backend == "none":
        return NullJudge()
    return Judge(
        build_chat_model(
            args.judge_backend,
            args.judge_model,
            GenerationParams(temperature=args.judge_temperature),
            base_url=args.judge_base_url,
        ),
        name=args.judge_model,
        delay=args.judge_delay,
    )


def _add_run_arguments(p: argparse.ArgumentParser) -> None:
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
        type=_int_list,
        default=(1, 2, 4),
        metavar="I,J,..",
        help="1-based questions answered directly by 'mixed' (default: 1,2,4)",
    )
    tools = g.add_mutually_exclusive_group()
    tools.add_argument("--tools", action="store_true", help="bind the calculator tool")
    tools.add_argument(
        "--tool-ablation",
        action="store_true",
        help="run every strategy both without and with tools",
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
    _add_judge_arguments(p)
    g = p.add_argument_group("data and execution")
    g.add_argument("--data-dir", type=Path, default=DATA_DIR / "cleaned")
    g.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    g.add_argument("--max-files", type=int, help="use only the first N traces (sorted)")
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


def build_parser() -> argparse.ArgumentParser:
    from netexplainer.evaluation import analysis

    parser = argparse.ArgumentParser(prog="netexplainer", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "download-data",
        help="download the captures of urls.txt, verified against hashes.txt",
    )
    p.add_argument("--urls", type=Path, help="list of URLs (default: urls.txt)")
    p.add_argument("--hashes", type=Path, help="SHA-256 list (default: hashes.txt)")
    p.add_argument("--no-verify", action="store_true", help="skip hash verification")
    p.add_argument(
        "--scrape",
        action="store_true",
        help="scrape the current Wireshark wiki instead (not reproducible)",
    )
    p = sub.add_parser("clean-data", help="keep captures with at most N packets")
    p.add_argument("max_packets", type=int)
    _add_run_arguments(sub.add_parser("run", help="run the evaluation"))
    p = sub.add_parser("rejudge", help="judge or re-judge the answers of a JSONL file")
    p.add_argument("results", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--max-attempts", type=int, default=10)
    _add_judge_arguments(p)
    analysis.build_parser(
        sub.add_parser("analyze", help="accuracy, CIs and McNemar tests")
    )
    p = sub.add_parser("convert-log", help="convert a legacy log to JSONL records")
    p.add_argument("log", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument(
        "--strategy", default="decomp", choices=["decomp", "mixed", "direct"]
    )
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--direct-questions", type=_int_list, default=(1, 2, 4))
    return parser


def cmd_run(args) -> int:
    from netexplainer.evaluation.runner import RunConfig, run_experiment

    models = args.models
    if not models:
        with open(args.questions) as f:
            models = yaml.safe_load(f).get("models") or []
    if not models:
        raise SystemExit("No models given (--models or 'models' in the questions file)")
    if "react" in args.strategy:
        raise SystemExit("The 'react' strategy is a placeholder, not implemented yet")
    judge = _build_judge(args)
    cfg = RunConfig(
        models=models,
        strategies=args.strategy,
        data_dir=args.data_dir,
        questions_path=args.questions,
        output=args.output_dir / "results.jsonl",
        tools=[False, True] if args.tool_ablation else [args.tools],
        backend=args.backend,
        base_url=args.base_url,
        params=GenerationParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            num_ctx=args.num_ctx,
            max_tokens=args.max_tokens,
        ),
        max_files=args.max_files,
        max_attempts=args.max_attempts,
        direct_questions=args.direct_questions,
        resume=args.resume,
        request_delay=args.request_delay,
    )
    written = run_experiment(cfg, judge)
    logger.info(f"Wrote {written} records to {cfg.output}")
    return 0


def cmd_convert_log(args) -> int:
    from netexplainer.evaluation.analysis import load_questions
    from netexplainer.evaluation.records import append_jsonl, parse_legacy_log

    questions = load_questions(args.questions)
    n = 0
    with open(args.log, encoding="utf-8") as f:
        for record in parse_legacy_log(
            f, args.strategy, questions, args.direct_questions
        ):
            append_jsonl(args.output, record)
            n += 1
    print(f"Wrote {n} records to {args.output}")
    return 0


def main(argv=None) -> int:
    from netexplainer.logger import configure_logger

    args = build_parser().parse_args(argv)
    configure_logger("netexplainer", args.log_file)
    load_dotenv()
    if args.command == "download-data":
        from netexplainer.scraper import Scraper

        if args.scrape:
            Scraper().download_captures()
            return 0
        from netexplainer.scraper import DEFAULT_HASHES, DEFAULT_URLS

        scraper = Scraper(scrape=False)
        report = scraper.download_listed(
            args.urls or DEFAULT_URLS,
            args.hashes or DEFAULT_HASHES,
            verify=not args.no_verify,
        )
        print({k: len(v) for k, v in report.items()})
        for k in ("mismatch", "missing_hash", "failed"):
            for name in report[k]:
                print(f"{k}: {name}")
        return 1 if report["mismatch"] or report["failed"] else 0
    elif args.command == "clean-data":
        from netexplainer.scraper import Scraper

        Scraper().clean_raw_data(max_packets=args.max_packets)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "analyze":
        from netexplainer.evaluation.analysis import run

        return run(args)
    elif args.command == "rejudge":
        from netexplainer.evaluation.rejudge import rejudge

        n = rejudge(args.results, args.output, _build_judge(args), args.max_attempts)
        print(f"Wrote {n} records to {args.output}")
    elif args.command == "convert-log":
        return cmd_convert_log(args)
    return 0
