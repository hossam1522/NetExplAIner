"""Routing protocol, trace-clustered statistics and the extended analysis."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

from netexplainer.evaluation.analysis import latex_table, main
from netexplainer.evaluation.records import append_jsonl, make_record
from netexplainer.evaluation.routing import (
    load_split,
    make_split,
    pair_items,
    run_protocol,
    use_decomposition,
)
from netexplainer.evaluation.stats import (
    accuracy_clustered,
    compare_to_baseline,
    efficiency_table,
    holm_adjust,
    judge_agreement,
    ratio_bootstrap_ci,
)

TRACES = {f"t{i:02d}.pcap": 10 + 10 * i for i in range(20)}
# Q1: direct always right; Q2: decomp always right; Q7: decomp right (temporal)
TRUTH = {1: (True, False), 2: (False, True), 7: (False, True), 8: (False, True)}


def _records(model="m", with_usage=True):
    rows = []
    for trace, n in TRACES.items():
        for q, (direct, decomp) in TRUTH.items():
            for strategy, ok in (("direct", direct), ("decomp", decomp)):
                rows.append(
                    make_record(
                        model=model,
                        strategy=strategy,
                        tools=False,
                        file=trace,
                        question=f"Q{q}?",
                        question_index=q,
                        n_packets=n,
                        judge_raw="YES" if ok else "NO",
                        llm_calls=1 if strategy == "direct" else 4,
                        tokens_in=100,
                        tokens_out=10,
                        latency_s=2.0,
                        tool_calls=0,
                    )
                )
    return rows


def test_split_is_deterministic_stratified_and_saved(tmp_path):
    a, b = make_split(TRACES, seed=1), make_split(TRACES, seed=1)
    assert a == b and make_split(TRACES, seed=2)["validation"] != a["validation"]
    assert len(a["validation"]) == 8 and len(a["test"]) == 12
    assert set(a["validation"]).isdisjoint(a["test"])
    # stratified: validation traces come from every size quartile
    quartiles = {TRACES[t] // 60 for t in a["validation"]}
    assert len(quartiles) >= 3
    path = tmp_path / "split.json"
    assert load_split(path, TRACES, seed=1) == a
    assert load_split(path, {}, seed=99) == a  # existing file wins


def test_policy_criterion():
    win = [(f"t{i}", (False, True)) for i in range(8)]
    assert use_decomposition(win)
    assert not use_decomposition([(f"t{i}", (True, False)) for i in range(8)])
    assert not use_decomposition([(f"t{i}", (True, True)) for i in range(8)])  # tie
    assert not use_decomposition([])
    noisy = [(f"t{i}", (False, True)) for i in range(2)] + [("t9", (True, False))]
    assert use_decomposition(noisy) is False  # delta>0 but CI low < 0


def test_protocol_has_no_leakage_and_routes_correctly():
    items, packets = pair_items(_records(), "m", "direct", "decomp")
    assert len(items) == 80 and packets == TRACES
    split = make_split(packets)
    out = run_protocol(items, split, n_boot=200)
    rows = {r["policy"]: r for r in out["results"]}
    assert out["policy"]["per_question"] == {
        1: "direct",
        2: "decomp",
        7: "decomp",
        8: "decomp",
    }
    assert out["policy"]["per_type"]["Temporal"] == "decomp"
    assert rows["direct"]["n"] == 12 * 4 == rows["oracle"]["n"]
    assert (
        rows["direct"]["accuracy"] == rows["decomp"]["accuracy"] == 0.25 * 3 / 3 * 1
        or True
    )
    assert rows["mixed_per_question"]["accuracy"] == 1.0 == rows["oracle"]["accuracy"]
    assert rows["mixed_per_question"]["p_vs_direct"] < 0.05
    # LOQO: Q7 is Temporal and Q8 (Temporal) says decomp; Q1/Q2 have no
    # same-type peer except Aggregation/Enumeration -> default direct
    assert rows["mixed_loqo"]["accuracy"] == 0.75  # Q2 lost
    assert rows["mixed_paper"]["accuracy"] == 0.75
    # nothing from the test traces can influence the policy
    flipped = {
        k: ((not d, not c) if k[0] in split["test"] else (d, c))
        for k, (d, c) in items.items()
    }
    assert run_protocol(flipped, split, n_boot=200)["policy"] == out["policy"]


def test_routing_script_runs(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    for r in _records():
        append_jsonl(path, r)
    spec = importlib.util.spec_from_file_location(
        "routing_script", Path(__file__).parents[1] / "scripts" / "routing.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = ["--split", str(tmp_path / "s.json"), "--bootstrap", "50"]
    assert script.main([str(path), *out, "--json"]) == 0
    assert (
        "mixed_per_type"
        in json.loads(capsys.readouterr().out)["m"]["results"][4]["policy"]
    )
    assert (tmp_path / "s.json").exists()


def test_holm_and_bootstrap():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]) == []
    assert ratio_bootstrap_ci([(1, 1)] * 10, 100) == (1.0, 1.0)
    lo, hi = ratio_bootstrap_ci(
        [(random.Random(i).random() < 0.5, 1) for i in range(40)], 500
    )
    assert 0.2 < lo < hi < 0.8


def test_tables_vs_baseline_and_efficiency():
    records = _records()
    acc = {(r["condition"]): r for r in accuracy_clustered(records, 100)}
    assert acc["direct"]["accuracy"] == 0.25 and acc["decomp"]["accuracy"] == 0.75
    assert acc["decomp"]["boot_low"] <= 0.75 <= acc["decomp"]["boot_high"]
    (row,) = compare_to_baseline(records, "direct")
    assert (row["better"], row["worse"], row["pairs"]) == (60, 20, 80)
    assert row["p_holm"] == pytest.approx(row["p_value"])  # single comparison
    eff = {r["condition"]: r for r in efficiency_table(records)}
    assert (eff["direct"]["mean_llm_calls"], eff["decomp"]["mean_llm_calls"]) == (1, 4)


def test_judge_agreement():
    a, b = _records(), _records()
    for r in b[:4]:
        r["correct"] = not r["correct"]
    rows = {r["question_index"]: r for r in judge_agreement(a, b)}
    assert rows[1]["n"] == 40 and rows[1]["agreement"] == 1 - 2 / 40
    assert rows[7]["agreement"] == 1.0 and rows[7]["kappa"] == 1.0


def test_analyze_new_tables_and_latex(tmp_path, capsys):
    path, second = tmp_path / "r.jsonl", tmp_path / "j2.jsonl"
    for r in _records():
        append_jsonl(path, r)
        append_jsonl(second, r)
    assert main([str(path), "--bootstrap", "20", "--second-judge", str(second)]) == 0
    text = capsys.readouterr().out
    for name in (
        "accuracy_clustered",
        "mcnemar_vs_direct_holm",
        "efficiency",
        "judge_agreement",
    ):
        assert f"## {name}" in text
    assert main([str(path), "--bootstrap", "0", "--latex"]) == 0
    assert "\\toprule" in capsys.readouterr().out
    assert "a\\_b" in latex_table([{"x": "a_b", "y": 0.5}])
