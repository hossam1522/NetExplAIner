"""Command line interface: ``python -m netexplainer <command> ...``."""

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from netexplainer import __version__
from netexplainer.cli_run import (
    DATA_DIR,
    DEFAULT_QUESTIONS,
    add_judge_arguments,
    add_run_arguments,
    build_judge,
    int_list,
)
from netexplainer.llm import GenerationParams

DEFAULT_LOG = DATA_DIR / "evaluation" / "netexplainer.log"

logger = logging.getLogger("netexplainer")


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
    p.add_argument("--raw-dir", type=Path, help="source directory (default: data/raw)")
    p.add_argument(
        "--cleaned-dir",
        type=Path,
        help="destination directory (default: data/cleaned); use another one "
        "per N, e.g. data/cleaned512, and pass it to run --data-dir",
    )
    add_run_arguments(sub.add_parser("run", help="run the evaluation"))
    p = sub.add_parser("rejudge", help="judge or re-judge the answers of a JSONL file")
    p.add_argument("results", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--max-attempts", type=int, default=10)
    add_judge_arguments(p)
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
    p.add_argument("--direct-questions", type=int_list, default=(1, 2, 4))
    return parser


def _tool_modes(args) -> list[str]:
    if args.tool_ablation:
        return ["none", "callable"]
    if args.tools_declared_not_callable:
        return ["declared"]
    return ["callable" if args.tools else "none"]


def cmd_run(args) -> int:
    from netexplainer.evaluation.runner import RunConfig, run_experiment

    models = args.models
    if not models:
        with open(args.questions) as f:
            models = yaml.safe_load(f).get("models") or []
    if not models:
        raise SystemExit("No models given (--models or 'models' in the questions file)")
    judge = build_judge(args)
    cfg = RunConfig(
        models=models,
        strategies=args.strategy,
        data_dir=args.data_dir,
        questions_path=args.questions,
        output=args.output_dir / "results.jsonl",
        tools=_tool_modes(args),
        question_set=args.question_set,
        max_packets=args.max_packets,
        min_packets=args.min_packets,
        react_max_steps=args.react_max_steps,
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

        kwargs = {}
        if args.raw_dir:
            kwargs["data_path"] = str(args.raw_dir)
        if args.cleaned_dir:
            kwargs["cleaned_path"] = str(args.cleaned_dir)
        Scraper().clean_raw_data(max_packets=args.max_packets, **kwargs)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "analyze":
        from netexplainer.evaluation.analysis import run

        return run(args)
    elif args.command == "rejudge":
        from netexplainer.evaluation.rejudge import rejudge

        n = rejudge(args.results, args.output, build_judge(args), args.max_attempts)
        print(f"Wrote {n} records to {args.output}")
    elif args.command == "convert-log":
        return cmd_convert_log(args)
    return 0
