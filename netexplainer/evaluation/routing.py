"""Leakage-free routing between ``direct`` and ``decomp`` (mixed strategy).

Protocol (the policy never sees the traces it is evaluated on):

a. Traces are split once, deterministically, into validation / test (default
   30% / 70%, stratified by number of packets, fixed seed); the lists are saved.
b. The policy is derived ONLY from validation: for a question (or a task type)
   use decomposition if ``delta = acc(decomp) - acc(direct) > 0`` and the lower
   end of the trace-clustered bootstrap CI of delta is ``>= 0`` (non-strict;
   ``strict=True`` asks ``> 0``). Ties and no evidence go to ``direct``.
c. The ``mixed`` answer of an item is its ``decomp`` answer when the policy says
   so and its ``direct`` answer otherwise (no new LLM calls are needed).
d. Leave-one-question-out: the policy of a held-out question is the policy of
   its task type derived from the validation items of the OTHER questions of
   that type (``direct`` if there are none).
e. Test accuracy of direct, decomp, the policies and the oracle (per question,
   best strategy on the test set itself: an optimistic upper bound), with Wilson
   and trace-bootstrap CIs and paired exact McNemar versus direct and decomp.
"""

import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from netexplainer.evaluation.metrics import mcnemar_exact, wilson_interval
from netexplainer.evaluation.stats import ratio_bootstrap_ci, trace_clusters

DIRECT, DECOMP = "direct", "decomp"
TEMPORAL, AGGREGATION, ENUMERATION, LOGIC = (
    "Temporal",
    "Aggregation",
    "Enumeration",
    "Logic&Retrieval",
)
# Task type of every question index (our classification of the 15 questions;
# override with a JSON {index: type} file).
TASK_TYPES = {
    1: AGGREGATION,
    2: ENUMERATION,
    3: LOGIC,
    4: AGGREGATION,
    5: AGGREGATION,
    6: LOGIC,
    7: TEMPORAL,
    8: TEMPORAL,
    9: TEMPORAL,
    10: LOGIC,
    11: ENUMERATION,
    12: AGGREGATION,
    13: LOGIC,
    14: LOGIC,
    15: AGGREGATION,
}

Items = dict[tuple[str, int], tuple[bool, bool]]  # (trace, q) -> (direct, decomp)


def make_split(
    packets: dict[str, int], val_frac: float = 0.3, seed: int = 0, strata: int = 4
) -> dict:
    """Stratified split of traces (``name -> n_packets``) into validation/test."""
    names = sorted(packets, key=lambda n: (packets[n], n))
    rng = random.Random(seed)
    validation: list[str] = []
    for i in range(strata):
        group = names[i * len(names) // strata : (i + 1) * len(names) // strata]
        rng.shuffle(group)
        validation += group[: round(val_frac * len(group))]
    test = sorted(set(names) - set(validation))
    return {
        "seed": seed,
        "val_frac": val_frac,
        "strata": strata,
        "validation": sorted(validation),
        "test": test,
    }


def load_split(path: Path, packets: dict[str, int], **kwargs) -> dict:
    """Read the split in ``path``; create and save it if it does not exist."""
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    split = make_split(packets, **kwargs)
    Path(path).write_text(json.dumps(split, indent=2) + "\n")
    return split


def pair_items(records: list[dict], model: str, direct: str, decomp: str):
    """Items answered with both strategies: ``Items`` and trace sizes."""
    by_cond: dict[str, dict] = {direct: {}, decomp: {}}
    packets: dict[str, int] = {}
    for r in records:
        if r["model"] == model and r["condition"] in by_cond:
            by_cond[r["condition"]][(r["file"], r["question_index"])] = (
                r.get("correct") is True
            )
            packets[r["file"]] = r.get("n_packets") or packets.get(r["file"], 0)
    common = by_cond[direct].keys() & by_cond[decomp].keys()
    items = {k: (by_cond[direct][k], by_cond[decomp][k]) for k in sorted(common)}
    return items, {f: packets[f] for f in {k[0] for k in items}}


def use_decomposition(
    items: list[tuple[str, tuple[bool, bool]]],
    n_boot: int = 1000,
    seed: int = 0,
    strict: bool = False,
) -> bool:
    """Policy criterion on ``(trace, (direct, decomp))`` validation items."""
    if not items:
        return False
    deltas = [(t, float(dec) - float(dr)) for t, (dr, dec) in items]
    mean = sum(d for _, d in deltas) / len(deltas)
    low, _ = ratio_bootstrap_ci(trace_clusters(deltas), n_boot, seed)
    return mean > 0 and (low > 0 if strict else low >= 0)


def derive_policies(
    items: Items, traces, types: dict[int, str], exclude=None, **kwargs
) -> tuple[dict, dict]:
    """(per-question policy, per-type policy) from the items of ``traces``.

    Questions in ``exclude`` are left out (leave-one-question-out). A policy
    maps question index / task type to ``True`` (decompose).
    """
    by_q, by_type = defaultdict(list), defaultdict(list)
    for (trace, q), pair in items.items():
        if trace in traces and q != exclude:
            by_q[q].append((trace, pair))
            by_type[types.get(q)].append((trace, pair))
    return (
        {q: use_decomposition(v, **kwargs) for q, v in by_q.items()},
        {t: use_decomposition(v, **kwargs) for t, v in by_type.items()},
    )


def evaluate(items: Items, traces, choose: Callable[[int], bool]) -> dict:
    """Outcome ``{(trace, q): correct}`` of routing each question with ``choose``."""
    return {(t, q): pair[choose(q)] for (t, q), pair in items.items() if t in traces}


def summarize(name: str, outcome: dict, ref: dict, n_boot: int, seed: int) -> dict:
    """Accuracy with Wilson/bootstrap CIs and McNemar versus ``ref`` strategies."""
    k, n = sum(outcome.values()), len(outcome)
    lo, hi = wilson_interval(k, n)
    blo, bhi = ratio_bootstrap_ci(
        trace_clusters((t, float(v)) for (t, _), v in outcome.items()), n_boot, seed
    )
    row = {
        "policy": name,
        "n": n,
        "accuracy": k / n if n else 0.0,
        "wilson_low": lo,
        "wilson_high": hi,
        "boot_low": blo,
        "boot_high": bhi,
    }
    for ref_name, other in ref.items():
        b = sum(1 for i in outcome if outcome[i] and not other[i])
        c = sum(1 for i in outcome if other[i] and not outcome[i])
        row[f"p_vs_{ref_name}"] = mcnemar_exact(b, c)
    return row


def run_protocol(
    items: Items,
    split: dict,
    types: dict[int, str] = TASK_TYPES,
    paper_direct: tuple[int, ...] = (1, 2, 4),
    n_boot: int = 1000,
    seed: int = 0,
    strict: bool = False,
) -> dict:
    """Steps b-e on ``items``: ``{"results": one row per policy on the test set,
    "policy": per-question and per-type choices derived from validation}``."""
    val, test = set(split["validation"]), set(split["test"])
    opts = {"n_boot": n_boot, "seed": seed, "strict": strict}
    per_q, per_type = derive_policies(items, val, types, **opts)
    loqo = {
        q: derive_policies(items, val, types, exclude=q, **opts)[1].get(types.get(q))
        for q in {q for _, q in items}
    }
    test_q = {
        q: [p for (t, qq), p in items.items() if t in test and qq == q] for q in loqo
    }
    # oracle: best strategy per question on the test set itself (ties -> direct)
    oracle = {q: sum(d for _, d in v) > sum(a for a, _ in v) for q, v in test_q.items()}
    policies = {
        "direct": lambda q: False,
        "decomp": lambda q: True,
        "mixed_paper": lambda q: q not in paper_direct,
        "mixed_per_question": lambda q: per_q.get(q, False),
        "mixed_per_type": lambda q: per_type.get(types.get(q), False),
        "mixed_loqo": lambda q: bool(loqo.get(q)),
        "oracle": lambda q: oracle[q],
    }
    outcomes = {n: evaluate(items, test, f) for n, f in policies.items()}
    ref = {DIRECT: outcomes["direct"], DECOMP: outcomes["decomp"]}
    name = {True: DECOMP, False: DIRECT}
    return {
        "results": [summarize(n, o, ref, n_boot, seed) for n, o in outcomes.items()],
        "policy": {
            "per_question": {q: name[v] for q, v in sorted(per_q.items())},
            "per_type": {str(t): name[v] for t, v in sorted(per_type.items(), key=str)},
        },
    }
