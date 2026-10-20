"""Analysis of result files: accuracy + Wilson CI and paired McNemar tests.

Usage: ``python -m netexplainer analyze results/results.jsonl [old.log ...]``
(or ``python scripts/analyze.py ...``). Legacy ``.log`` files are parsed with
:func:`netexplainer.evaluation.records.parse_legacy_log`.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from netexplainer.evaluation.metrics import accuracy_table, paired_tests
from netexplainer.evaluation.records import load_records
from netexplainer.evaluation.stats import (
    accuracy_clustered,
    compare_to_baseline,
    efficiency_table,
    judge_agreement,
)

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[1] / "data" / "questions.yaml"


def load_questions(path: Path) -> list[str]:
    with open(path) as f:
        return [item["question"] for item in yaml.safe_load(f)["questions"]]


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "(no data)\n"

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict]) -> str:
    """``tabular`` with the columns of ``rows`` (3 decimals, escaped text)."""
    if not rows:
        return "% (no data)\n"

    def fmt(v):
        text = f"{v:.3f}" if isinstance(v, float) else str(v)
        for ch in "&%_#$":
            text = text.replace(ch, "\\" + ch)
        return text

    cols = list(rows[0])
    lines = [
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        " & ".join(fmt(c) for c in cols) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(fmt(r[c]) for c in cols) + " \\\\" for r in rows]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def similarity_table(records: list[dict]) -> list[dict]:
    """Mean judge similarity of generated sub-questions per model/condition."""
    groups = defaultdict(list)
    for r in records:
        if r.get("subq_similarity") is not None:
            groups[(r["model"], r["condition"])].append(r["subq_similarity"])
    return [
        {"model": m, "condition": c, "n": len(v), "mean_similarity": sum(v) / len(v)}
        for (m, c), v in sorted(groups.items())
    ]


def build_parser(parser: argparse.ArgumentParser | None = None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="+", type=Path, help=".jsonl or legacy .log files"
    )
    parser.add_argument(
        "--legacy-strategy",
        default="decomp",
        choices=["decomp", "mixed", "direct"],
        help="strategy to assign to records parsed from legacy .log files",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--by-question", action="store_true", help="also per question")
    parser.add_argument(
        "--plots",
        type=Path,
        metavar="DIR",
        help="write charts (needs the 'plots' extra)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print JSON instead of markdown"
    )
    parser.add_argument(
        "--latex", action="store_true", help="print LaTeX tabulars (booktabs)"
    )
    parser.add_argument(
        "--baseline",
        default="direct",
        help="condition of the McNemar tests with Holm correction (default: direct)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        metavar="B",
        help="trace-clustered bootstrap replicates (0 = skip; default 1000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument(
        "--second-judge",
        type=Path,
        metavar="JSONL",
        help="the same answers judged by another judge (see 'rejudge'): "
        "agreement per question",
    )
    return parser


def run(args) -> int:
    questions = load_questions(args.questions) if args.questions.exists() else None
    records = load_records(args.results, args.legacy_strategy, questions)
    tables = {"accuracy": accuracy_table(records, ("model", "condition"))}
    if args.by_question:
        tables["accuracy_by_question"] = accuracy_table(
            records, ("model", "condition", "question_index", "question")
        )
    tables["subquestion_similarity"] = similarity_table(records)
    tables["mcnemar"] = paired_tests(records)
    tables["accuracy_clustered"] = accuracy_clustered(
        records, args.bootstrap, args.seed
    )
    tables[f"mcnemar_vs_{args.baseline}_holm"] = compare_to_baseline(
        records, args.baseline
    )
    tables["efficiency"] = efficiency_table(records)
    if args.second_judge:
        tables["judge_agreement"] = judge_agreement(
            records, load_records([args.second_judge])
        )
    if args.json:
        print(json.dumps(tables, indent=2, default=str))
    else:
        render = latex_table if args.latex else markdown_table
        for name, rows in tables.items():
            print(f"## {name}\n\n{render(rows)}")
    if args.plots:
        from netexplainer.evaluation.plots import generate_plots

        generate_plots(records, args.plots)
    return 0


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))
