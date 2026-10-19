"""Accuracy with Wilson confidence intervals and paired McNemar tests.

An item counts as correct only if the judge said YES; NO and PROBLEM (judge
or pipeline failure) count as incorrect, as in the paper's pie charts.
"""

import math
from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% by default)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact (binomial) two-sided McNemar p-value from the discordant counts.

    b: pairs where A is correct and B is not; c: B correct and A not.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def accuracy_table(records: Iterable[dict], by: tuple[str, ...]) -> list[dict]:
    """Accuracy per group of the ``by`` fields, with 95% Wilson CI."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[tuple(r.get(k) for k in by)].append(r)
    rows = []
    for key in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        items = groups[key]
        n = len(items)
        k = sum(1 for r in items if r.get("correct") is True)
        problems = sum(1 for r in items if r.get("correct") is None)
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                **dict(zip(by, key)),
                "n": n,
                "correct": k,
                "problems": problems,
                "accuracy": k / n if n else 0.0,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return rows


def paired_tests(records: Iterable[dict]) -> list[dict]:
    """McNemar test between every pair of conditions of the same model.

    Items are paired on (model, file, question); only items present in both
    conditions are used.
    """
    outcome: dict[tuple, dict[tuple, bool]] = defaultdict(dict)
    for r in records:
        outcome[(r["model"], r["condition"])][(r["file"], r["question"])] = (
            r.get("correct") is True
        )
    rows = []
    models = sorted({m for m, _ in outcome})
    for model in models:
        conds = sorted(c for m, c in outcome if m == model)
        for a, b in combinations(conds, 2):
            xa, xb = outcome[(model, a)], outcome[(model, b)]
            common = xa.keys() & xb.keys()
            n_ab = sum(1 for i in common if xa[i] and not xb[i])
            n_ba = sum(1 for i in common if xb[i] and not xa[i])
            rows.append(
                {
                    "model": model,
                    "a": a,
                    "b": b,
                    "pairs": len(common),
                    "acc_a": sum(xa[i] for i in common) / len(common)
                    if common
                    else 0.0,
                    "acc_b": sum(xb[i] for i in common) / len(common)
                    if common
                    else 0.0,
                    "a_only": n_ab,
                    "b_only": n_ba,
                    "p_value": mcnemar_exact(n_ab, n_ba),
                }
            )
    return rows
