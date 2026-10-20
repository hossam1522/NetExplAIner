"""Statistics for the paper tables: trace-clustered bootstrap, McNemar versus a
baseline with Holm correction, efficiency and agreement between two judges.

Questions of the same trace are not independent, so besides the Wilson interval
(which treats every (trace, question) as independent) accuracy gets a
percentile bootstrap that resamples whole traces (clusters).
"""

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from netexplainer.evaluation.metrics import mcnemar_exact, wilson_interval
from netexplainer.evaluation.records import record_key


def ratio_bootstrap_ci(
    clusters: Sequence[tuple[float, int]],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of ``sum(x) / sum(n)`` resampling ``(x, n)`` clusters."""
    if not clusters or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choices(clusters, k=len(clusters))
        n = sum(c[1] for c in sample)
        stats.append(sum(c[0] for c in sample) / n if n else 0.0)
    stats.sort()
    lo = stats[int(n_boot * alpha / 2)]
    hi = stats[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]
    return (lo, hi)


def trace_clusters(items: Iterable[tuple[str, float]]) -> list[tuple[float, int]]:
    """Group ``(trace, value)`` items into ``(sum of values, count)`` clusters."""
    groups: dict[str, list[float]] = defaultdict(list)
    for trace, value in items:
        groups[trace].append(value)
    return [(sum(v), len(v)) for _, v in sorted(groups.items())]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _outcomes(records: Iterable[dict]) -> dict[tuple, dict[tuple, bool]]:
    """(model, condition) -> {(file, question): correct}."""
    out: dict[tuple, dict[tuple, bool]] = defaultdict(dict)
    for r in records:
        out[(r["model"], r["condition"])][(r["file"], r["question"])] = (
            r.get("correct") is True
        )
    return out


def accuracy_clustered(records: list[dict], n_boot: int = 2000, seed: int = 0):
    """Accuracy per model x condition: Wilson CI and trace-bootstrap CI."""
    rows = []
    for (model, cond), res in sorted(_outcomes(records).items()):
        k, n = sum(res.values()), len(res)
        lo, hi = wilson_interval(k, n)
        blo, bhi = ratio_bootstrap_ci(
            trace_clusters((f, float(v)) for (f, _), v in res.items()), n_boot, seed
        )
        rows.append(
            {
                "model": model,
                "condition": cond,
                "n": n,
                "accuracy": k / n if n else 0.0,
                "wilson_low": lo,
                "wilson_high": hi,
                "boot_low": blo,
                "boot_high": bhi,
            }
        )
    return rows


def compare_to_baseline(records: list[dict], baseline: str = "direct") -> list[dict]:
    """Exact McNemar of every condition against ``baseline`` of the same model,
    paired on (trace, question), with Holm correction over all comparisons."""
    out = _outcomes(records)
    rows = []
    for (model, cond), res in sorted(out.items()):
        base = out.get((model, baseline))
        if cond == baseline or base is None:
            continue
        common = res.keys() & base.keys()
        b = sum(1 for i in common if res[i] and not base[i])
        c = sum(1 for i in common if base[i] and not res[i])
        rows.append(
            {
                "model": model,
                "condition": cond,
                "pairs": len(common),
                "acc_baseline": sum(base[i] for i in common) / len(common)
                if common
                else 0.0,
                "acc_condition": sum(res[i] for i in common) / len(common)
                if common
                else 0.0,
                "better": b,
                "worse": c,
                "p_value": mcnemar_exact(b, c),
            }
        )
    for row, p in zip(rows, holm_adjust([r["p_value"] for r in rows])):
        row["p_holm"] = p
    return rows


_USAGE = ("llm_calls", "tokens_in", "tokens_out", "latency_s", "tool_calls")


def efficiency_table(records: Iterable[dict]) -> list[dict]:
    """Mean LLM calls, tokens, latency and tool calls per model x condition
    (records without usage, e.g. legacy logs, are ignored)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("llm_calls") is not None:
            groups[(r["model"], r["condition"])].append(r)
    return [
        {"model": m, "condition": c, "n": len(rs)}
        | {f"mean_{k}": sum(r.get(k) or 0 for r in rs) / len(rs) for k in _USAGE}
        for (m, c), rs in sorted(groups.items())
    ]


def judge_agreement(records_a: list[dict], records_b: list[dict]) -> list[dict]:
    """Agreement (raw and Cohen's kappa) of two judges per question, over the
    answers (same ``record_key``) judged by both."""
    b_by_key = {record_key(r): r.get("correct") is True for r in records_b}
    groups: dict[tuple, list[tuple[bool, bool]]] = defaultdict(list)
    for r in records_a:
        key = record_key(r)
        if key in b_by_key:
            groups[(r.get("question_index"), r["question"])].append(
                (r.get("correct") is True, b_by_key[key])
            )
    rows = []
    for (idx, question), pairs in sorted(groups.items(), key=lambda x: str(x[0])):
        n = len(pairs)
        po = sum(a == b for a, b in pairs) / n
        pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        rows.append(
            {
                "question_index": idx,
                "question": question,
                "n": n,
                "agreement": po,
                "kappa": kappa,
            }
        )
    return rows
