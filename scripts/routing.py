"""Leakage-free routing protocol on existing direct and decomp results.

    python scripts/routing.py results/a/results.jsonl results/b/results.jsonl \
        --split results/split.json [--model phi4-q4] [--json]

See ``netexplainer/evaluation/routing.py`` for the protocol.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netexplainer.evaluation.analysis import markdown_table  # noqa: E402
from netexplainer.evaluation.records import load_records  # noqa: E402
from netexplainer.evaluation.routing import (  # noqa: E402
    TASK_TYPES,
    load_split,
    pair_items,
    run_protocol,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+", type=Path, help=".jsonl with direct + decomp")
    p.add_argument("--split", type=Path, default=Path("split.json"))
    p.add_argument("--model", help="only this model (default: every model)")
    p.add_argument("--direct", default="direct", help="condition of direct answers")
    p.add_argument("--decomp", default="decomp", help="condition of decomp answers")
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=1000, metavar="B")
    p.add_argument("--strict", action="store_true", help="CI low > 0 (not >= 0)")
    p.add_argument(
        "--paper-direct", default="1,2,4", help="mixed_paper direct questions"
    )
    p.add_argument("--task-types", type=Path, help="JSON {question index: type}")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    types = TASK_TYPES
    if args.task_types:
        types = {int(k): v for k, v in json.loads(args.task_types.read_text()).items()}
    records = load_records(args.results)
    models = [args.model] if args.model else sorted({r["model"] for r in records})
    report = {}
    for model in models:
        items, packets = pair_items(records, model, args.direct, args.decomp)
        if not items:
            continue
        # one split for all models: shared traces are split by name and size
        split = load_split(args.split, packets, val_frac=args.val_frac, seed=args.seed)
        report[model] = run_protocol(
            items,
            split,
            types,
            tuple(int(x) for x in args.paper_direct.split(",") if x),
            args.bootstrap,
            args.seed,
            args.strict,
        )
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    for model, out in report.items():
        print(f"## {model}: test accuracy of the routing policies\n")
        print(markdown_table(out["results"]))
        print(f"policy from validation: {json.dumps(out['policy'])}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
