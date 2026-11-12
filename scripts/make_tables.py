"""Tables for the reviewers' response from results/judged_final (see TABLES.md).

    .venv/bin/python scripts/make_tables.py  -> results/tables/*.md, *.csv
Reuses netexplainer.evaluation.{metrics,stats,routing,analysis}.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netexplainer.evaluation.analysis import markdown_table  # noqa: E402
from netexplainer.evaluation.metrics import accuracy_table  # noqa: E402
from netexplainer.evaluation.records import read_jsonl  # noqa: E402
from netexplainer.evaluation.routing import (  # noqa: E402
    TASK_TYPES, derive_policies, load_split, pair_items, run_protocol)
from netexplainer.evaluation.stats import (  # noqa: E402
    _outcomes, accuracy_clustered, compare_to_baseline, efficiency_table,
    holm_adjust, judge_agreement, ratio_bootstrap_ci, trace_clusters)

R = Path("/home/hossam/Projects/NetExplAIner/results")
OUT = R / "tables"
SUB = R / "campaign/subsets"
S160 = set((SUB / "s160_names.txt").read_text().split())
PAPER_DIRECT = (1, 2, 4)
BOOT = 2000


def save(name, rows, title):
    if not rows:
        return
    rows = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows]
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / f"{name}.md").write_text(f"## {title}\n\n{markdown_table(rows)}")


def pct(rows):
    """Proportions -> percentages (accuracy/CI/acc_ fields)."""
    out = []
    for r in rows:
        r = dict(r)
        for k, v in r.items():
            if isinstance(v, float) and k.startswith(("accuracy", "wilson", "boot", "acc_", "ci_", "diff")):
                r[k] = round(100 * v, 2)
        out.append(r)
    return out


def load(group):
    return [r for p in sorted((R / "judged_final" / group).glob("*/results.jsonl"))
            for r in read_jsonl(p)]


def derive_mixed(records):
    """mixed[+tools|+declared]: direct for Q1,Q2,Q4, decomp[...] for the rest."""
    by = defaultdict(dict)
    for r in records:
        by[(r["model"], r["file"], r["question_index"])][r["condition"]] = r
    out = []
    for (_, _, q), d in by.items():
        for dec, name in (("decomp", "mixed"), ("decomp+tools", "mixed+tools"),
                          ("decomp+declared", "mixed+declared")):
            src = d.get("direct") if q in PAPER_DIRECT else d.get(dec)
            if dec in d and "direct" in d and src is not None:
                out.append({**src, "strategy": "mixed", "condition": name})
    return out


def with_ci(rows, records, baseline):
    """compare_to_baseline + diff (pp) and trace-bootstrap CI of the paired diff."""
    out = _outcomes(records)
    for r in rows:
        a, b = out[(r["model"], r["condition"])], out[(r["model"], baseline)]
        common = sorted(a.keys() & b.keys())
        lo, hi = ratio_bootstrap_ci(trace_clusters(
            (f, float(a[(f, q)]) - float(b[(f, q)])) for f, q in common), BOOT)
        r["diff_pp"], r["diff_low"], r["diff_high"] = (
            r["acc_condition"] - r["acc_baseline"], lo, hi)
    return rows


def mcnemar_table(records, baseline="direct", only=None):
    rows = [r for r in compare_to_baseline(records, baseline)
            if only is None or r["condition"] in only]
    for r, p in zip(rows, holm_adjust([r["p_value"] for r in rows])):
        r["p_holm"] = p
    return pct(with_ci(rows, records, baseline))


def traces(records):
    n = defaultdict(set)
    for r in records:
        n[(r["model"], r["condition"])].add(r["file"])
    return n


def main():
    OUT.mkdir(exist_ok=True)
    main318, qwen, ext160, ext80 = (load(g) for g in ("main", "main_s160", "ext_s160", "ext_s80"))
    t0, t0s = load("temp0"), load("temp0_s160")
    all318 = main318 + derive_mixed(main318)
    s160 = [r for r in main318 if r["file"] in S160] + qwen
    s160 += derive_mixed(s160)
    CONDS = ["direct", "cot", "decomp", "decomp+declared", "decomp+tools", "mixed",
             "mixed+declared", "mixed+tools"]
    rank = {c: i for i, c in enumerate(CONDS)}

    def srt(rows):
        return sorted(rows, key=lambda r: (r["model"], rank.get(r["condition"], 99)))

    # 0 inventory
    inv = []
    for g in ("main", "main_s160", "ext_s160", "ext_s80", "temp0", "temp0_s160"):
        recs = load(g)
        for (m, c), fs in sorted(traces(recs).items()):
            n = sum(1 for r in recs if (r["model"], r["condition"]) == (m, c))
            inv.append({"group": g, "model": m, "condition": c, "n_items": n, "n_traces": len(fs)})
    save("t0_inventory", inv, "Cells found in judged_final")

    # 1 accuracy
    for tag, recs, title in (("318", all318, "318 traces"), ("s160", s160, "160-trace subset (all models)")):
        acc = accuracy_clustered(recs, BOOT)
        nt = traces(recs)
        for r in acc:
            r["n_traces"] = len(nt.get((r["model"], r["condition"]), ())) or None
        save(f"t1_accuracy_{tag}", pct(srt(acc)),
             f"Accuracy (%) per model x condition, {title}; Wilson and trace-bootstrap 95% CI")
        # 2 McNemar vs direct
        save(f"t2_mcnemar_vs_direct_{tag}", srt(mcnemar_table(recs)),
             f"Exact McNemar vs direct (Holm over this table), {title}; diff in pp with trace-bootstrap CI")
        # 3 mixed
        mixed = [r for r in pct(acc) if r["condition"].startswith(("mixed", "direct", "decomp"))]
        save(f"t3_mixed_{tag}", srt(mixed), f"Mixed (Q1,Q2,Q4 direct, rest decomp) vs direct/decomp, {title}")
        fam = []
        for dec, mx in (("decomp", "mixed"), ("decomp+tools", "mixed+tools"),
                        ("decomp+declared", "mixed+declared")):
            fam += mcnemar_table([r for r in recs if r["condition"] in (dec, mx)], dec, only={mx})
        save(f"t3_mixed_mcnemar_vs_decomp_{tag}", srt(fam),
             f"McNemar of mixed[+tools|+declared] vs its own decomp[+tools|+declared], {tag}")

    # 5 per question
    for tag, recs in (("core_318", all318), ("core_s160", s160), ("ext_phi4_s160", ext160),
                      ("ext_qwen_s80", ext80)):
        byq = accuracy_table(recs, ("model", "condition", "question_index"))
        wide = defaultdict(dict)
        for r in byq:
            wide[(r["model"], r["condition"])][f"Q{r['question_index']}"] = (
                f"{100 * r['accuracy']:.1f} (n={r['n']})" if "ext" in tag else round(100 * r["accuracy"], 1))
        rows = [{"model": m, "condition": c, **dict(sorted(v.items(), key=lambda x: int(x[0][1:])))}
                for (m, c), v in wide.items()]
        save(f"t5_per_question_{tag}", srt(rows), f"Accuracy (%) per question, {tag}")
    for tag, recs in (("ext_phi4_s160", ext160), ("ext_qwen_s80", ext80)):
        acc = accuracy_clustered(recs, BOOT)
        save(f"t5_ext_accuracy_{tag}", pct(srt(acc)), f"Extended Q10-Q15 overall accuracy (%), {tag}")
        save(f"t5_ext_mcnemar_{tag}", srt(mcnemar_table(recs)), f"Extended Q10-Q15 McNemar vs direct, {tag}")

    # 6 efficiency
    for tag, recs in (("318", all318), ("s160", s160)):
        eff = efficiency_table(recs)
        base = {r["model"]: r for r in eff if r["condition"] == "direct"}
        for r in eff:
            b = base.get(r["model"])
            r["calls_x_direct"] = r["mean_llm_calls"] / b["mean_llm_calls"] if b else None
            r["tokens_out_x_direct"] = r["mean_tokens_out"] / b["mean_tokens_out"] if b else None
            r["latency_x_direct"] = r["mean_latency_s"] / b["mean_latency_s"] if b else None
        save(f"t6_efficiency_{tag}", srt(eff), f"Mean cost per item, {tag}")
    g = defaultdict(list)
    for r in main318 + qwen:
        g[(r["model"], r["condition"], r["question_index"])].append(r)
    rows = [{"model": m, "condition": c, "question": q, "n": len(v),
             **{f"mean_{k}": sum(x.get(k) or 0 for x in v) / len(v)
                for k in ("llm_calls", "tokens_in", "tokens_out", "latency_s", "tool_calls")}}
            for (m, c, q), v in sorted(g.items()) if rank.get(c, 99) < 99]
    save("t6_efficiency_per_question", rows, "Mean calls/tokens/latency per question (core)")

    # 7 temperature
    def tag_t(recs, t):
        return [{**r, "condition": f"{r['condition']}@T{t}"} for r in recs]
    s_main = [r for r in main318 if r["file"] in S160]
    rows = []
    for model, cond, hot, cold, label in (
        ("cogito-8b", "decomp", main318, t0, "318 traces"),
        ("phi4-q4", "decomp", s_main, t0s, "s160"),
        ("phi4-q4", "direct", s_main, t0s, "s160")):
        recs = [r for r in tag_t(hot, "0.8") + tag_t(cold, "0") if r["model"] == model
                and r["condition"].startswith(cond + "@")]
        for r in mcnemar_table(recs, f"{cond}@T0.8"):
            rows.append({"set": label, **r})
    save("t7_temperature", rows, "Temperature 0 vs 0.8 (same traces), exact McNemar paired (condition = T0)")
    ta = accuracy_clustered([r for r in tag_t(s_main, "0.8") if r["model"] == "phi4-q4"
                             and r["condition"] in ("direct@T0.8", "decomp@T0.8")] + tag_t(t0s, "0") + [
        r for r in tag_t(main318, "0.8") + tag_t(t0, "0") if r["model"] == "cogito-8b"
        and r["condition"].startswith("decomp@")], BOOT)
    save("t7_temperature_accuracy", pct(ta), "Accuracy (%) by temperature (phi4: s160 subset; cogito: 318)")

    # 8 tool ablation (all models with several tool modes)
    abl = [r for r in main318 + qwen if r["condition"] in ("direct", "decomp", "decomp+declared", "decomp+tools")
           and r["model"] in ("cogito-8b", "mistral-7b", "qwen2.5-32b")]
    abl_q = [r for r in s160 if r["model"] == "qwen2.5-32b" and r["condition"] in
             ("direct", "decomp", "decomp+tools")]
    abl = [r for r in abl if r["model"] != "qwen2.5-32b"] + abl_q
    save("t8_tools_vs_direct", srt(mcnemar_table(abl)), "Tool ablation: McNemar vs direct (Holm)")
    save("t8_tools_vs_decomp", srt(mcnemar_table(
        [r for r in abl if r["condition"] != "direct"], "decomp")), "Tool ablation: McNemar vs decomp (none)")
    use = []
    for (m, c), _ in sorted(traces(abl).items()):
        v = [r for r in abl if (r["model"], r["condition"]) == (m, c)]
        if c.startswith("decomp"):
            use.append({"model": m, "condition": c, "n": len(v),
                        "tool_calls_total": sum(r.get("tool_calls") or 0 for r in v),
                        "items_with_call": sum(1 for r in v if (r.get("tool_calls") or 0) > 0),
                        "blocked_total": sum(r.get("tool_calls_blocked") or 0 for r in v)})
    save("t8_tool_usage", use, "Tool invocations per cell")

    # 4 routing
    packets = {}
    for r in main318:
        packets[r["file"]] = r["n_packets"]
    split = load_split(OUT / "split_30_70.json", packets)
    rrows, prows, srows = [], [], []
    cfgs = [(m, d) for m in ("cogito-8b", "mistral-7b", "phi4-q4", "llama2-7b", "qwen2.5-32b")
            for d in ("decomp", "decomp+tools")]
    for model, dec in cfgs:
        recs = s160 if model == "qwen2.5-32b" else main318
        items, _ = pair_items(recs, model, "direct", dec)
        if not items:
            continue
        val = set(split["validation"])
        out = run_protocol(items, split, n_boot=1000)
        for r in out["results"]:
            rrows.append({"model": model, "decomp_cond": dec, "n_test_items": r["n"],
                          **{k: v for k, v in r.items() if k not in ("n",)}})
        pq = out["policy"]["per_question"]
        loqo = {q: derive_policies(items, val, TASK_TYPES, exclude=q, n_boot=1000)[1].get(TASK_TYPES[q])
                for q in sorted({q for _, q in items})}
        dset = lambda f: sorted(q for q in f if not f[q])
        prows.append({"model": model, "decomp_cond": dec,
                      "direct_questions_per_question_policy": dset({int(q): v == "decomp" for q, v in pq.items()}),
                      "direct_questions_loqo": dset({q: bool(v) for q, v in loqo.items()}),
                      "per_type_policy": json.dumps(out["policy"]["per_type"]),
                      "matches_Q1_Q2_Q4_perq": dset({int(q): v == "decomp" for q, v in pq.items()}) == [1, 2, 4],
                      "matches_Q1_Q2_Q4_loqo": dset({q: bool(v) for q, v in loqo.items()}) == [1, 2, 4]})
        accs = defaultdict(list)
        for seed in range(10):
            sp = {**split, **__import__("netexplainer.evaluation.routing", fromlist=["x"]).make_split(
                packets, seed=seed)}
            for r in run_protocol(items, sp, n_boot=100, seed=seed)["results"]:
                accs[r["policy"]].append(r["accuracy"])
        srows.append({"model": model, "decomp_cond": dec,
                      **{k: f"{100 * sum(v) / len(v):.1f}±{100 * (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** .5:.1f}"
                         for k, v in accs.items()}})
    save("t4_routing_test", pct(rrows), "Routing, split 30/70 (seed 0): test accuracy (%) of learned policies vs oracle/baselines")
    save("t4_routing_policies", prows, "Policies learned on validation (questions routed to direct) vs paper (Q1,Q2,Q4)")
    save("t4_routing_10_splits", srows, "Test accuracy (%) mean±sd over 10 random 30/70 splits (bootstrap 100)")

    # 9 paper comparison
    paper = [r for r in read_jsonl(R / "replication/original.jsonl")]
    pacc = {(r["model"], r["condition"]): r for r in accuracy_table(paper, ("model", "condition"))}
    t301 = {r["file"] for r in paper if r["model"] == "phi4" and r["condition"] == "decomp"}
    aliases = {"phi4": "phi4-q4"}
    sub301 = [r for r in all318 if r["file"] in t301]
    a318 = {(r["model"], r["condition"]): r for r in accuracy_clustered(all318, 200)}
    a301 = {(r["model"], r["condition"]): r for r in accuracy_clustered(sub301, 200)}
    nopaper = {("mistral-7b", "direct"): (33.56, "earlier study"), ("llama2-7b", "direct"): (18.74, "earlier study"),
               ("llama2-7b", "decomp"): (16.46, "earlier study")}
    rows = []
    for (m, c) in sorted(a318, key=lambda k: (k[0], rank.get(k[1], 99))):
        pm = next((k for k, v in aliases.items() if v == m), m)
        p = pacc.get((pm, c))
        pv = 100 * p["accuracy"] if p else nopaper.get((m, c), (None,))[0]
        d = 100 * (a318[(m, c)]["accuracy"] - a301[(m, c)]["accuracy"])
        rows.append({"model": m, "condition": c, "paper_acc": pv, "paper_n": p["n"] if p else None,
                     "new_318": 100 * a318[(m, c)]["accuracy"], "new_301": 100 * a301[(m, c)]["accuracy"],
                     "delta_318_vs_301_pp": d, "moves_gt_1pp": abs(d) > 1,
                     "new318_minus_paper_pp": None if pv is None else 100 * a318[(m, c)]["accuracy"] - pv})
    save("t9_table5_vs_paper", rows, "Table 5: new runs (gemma3:27b local judge) vs paper (API judge); 318 vs the paper's 301 traces")
    qp = {"decomp": 53.08, "decomp+tools": 57.22, "mixed+tools": 65.82}
    save("t9_table5_s160_qwen", [{**r, "paper_acc_301": qp.get(r["condition"])} for r in
                                 pct(srt(accuracy_clustered(s160, 200))) if r["model"] == "qwen2.5-32b"],
         "Qwen2.5-32B (s160) for Table 5")

    # 10 judge agreement
    old = [{**r, "model": "phi4"} for r in paper if r["model"] == "phi4-fp16" and r["condition"] == "mixed"]
    new = list(read_jsonl(R / "replication/data/orig_phi4fp16_full_rejudged_n.jsonl"))
    ja = judge_agreement(old, new)
    save("t10_judge_agreement", ja, "API judge (paper, phi4-fp16 mixed) vs gemma3:27b local on the same 2862 answers, per question")


main()
