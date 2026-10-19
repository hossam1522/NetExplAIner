import json
from types import SimpleNamespace

import pytest

from netexplainer.cli import main
from netexplainer.evaluation.judge import Judge, parse_similarity, parse_verdict
from netexplainer.evaluation.metrics import (
    accuracy_table,
    mcnemar_exact,
    paired_tests,
    wilson_interval,
)
from netexplainer.evaluation.records import (
    completed_keys,
    make_record,
    parse_legacy_log,
    read_jsonl,
)
from netexplainer.evaluation.runner import RunConfig, conditions, run_experiment

QUESTIONS = [
    "Q one?",
    "What predominates in the capture: ICMP, TCP, or UDP?",
    "Q three?",
]


# --- judge ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("YES", "YES"),
        ("NO\n", "NO"),
        ("**Yes.**", "YES"),
        ("no", "NO"),
        ("Maybe", "PROBLEM"),
        ("NOTE: yes", "PROBLEM"),
        (None, "PROBLEM"),
    ],
)
def test_parse_verdict(raw, expected):
    assert parse_verdict(raw) == expected


def test_parse_similarity():
    assert parse_similarity("80%") == 80.0
    assert parse_similarity(" 75 \n") == 75.0
    assert parse_similarity("high") is None


def test_judge_prompts(fake_chat_model):
    model = fake_chat_model(lambda m: "YES")
    sleeps = []
    judge = Judge(model, delay=0.5, sleep=sleeps.append)
    assert judge.evaluate_answer("Q?", "42", 42) == "YES"
    assert judge.evaluate_subquestions("Q?", ["a"], ["b"]) == "YES"
    assert sleeps == [0.5, 0.5]
    assert 'Answer LLM: "42"' in model.calls[0][0].content
    assert "Subquestions: \"['a']\"" in model.calls[1][0].content


# --- metrics ----------------------------------------------------------------


def test_wilson_interval():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)


def _rec(model, condition, file, question, correct):
    return {
        "model": model,
        "condition": condition,
        "file": file,
        "question": question,
        "question_index": 1,
        "correct": correct,
    }


def test_accuracy_and_paired_tests():
    records = [
        _rec("m", "decomp", "f1", "q", True),
        _rec("m", "decomp", "f2", "q", True),
        _rec("m", "direct", "f1", "q", False),
        _rec("m", "direct", "f2", "q", None),
        _rec("m", "direct", "f3", "q", True),
    ]
    rows = {r["condition"]: r for r in accuracy_table(records, ("model", "condition"))}
    assert rows["decomp"]["accuracy"] == 1.0
    assert (
        rows["direct"]["n"],
        rows["direct"]["correct"],
        rows["direct"]["problems"],
    ) == (3, 1, 1)
    (test,) = paired_tests(records)
    assert (test["a"], test["b"], test["pairs"]) == ("decomp", "direct", 2)
    assert (test["a_only"], test["b_only"]) == (2, 0)


# --- records ----------------------------------------------------------------

LEGACY_LOG = """\
2025-12-01 10:00:00 - evaluator - INFO - Model: phi4, File: a.pcap, Question: Q one?, Subquestions Eval: 100, Answer Eval: YES
2025-12-01 10:00:01 - evaluator - DEBUG - something else
2025-12-01 10:00:02 - evaluator - INFO - Model: phi4_tools, File: b.pcap, Question: What predominates in the capture: ICMP, TCP, or UDP?, Subquestions Eval: 80%, Answer Eval: NO
2025-12-01 10:00:03 - evaluator - INFO - Model: phi4, Question: Q one?, Similarity: 90.0
2025-12-01 10:00:04 - evaluator - INFO - Model: phi4, File: c.pcap, Question: Q three?, Subquestions Eval: ERROR, Answer Eval: PROBLEM
"""


def test_parse_legacy_log():
    recs = list(parse_legacy_log(LEGACY_LOG.splitlines(), "mixed", QUESTIONS, (1,)))
    assert len(recs) == 3
    r0, r1, r2 = recs
    assert (r0["model"], r0["tools"], r0["decomposed"], r0["subq_similarity"]) == (
        "phi4",
        False,
        False,
        None,
    )
    assert r0["correct"] is True and r0["condition"] == "mixed"
    assert r1["question"] == QUESTIONS[1] and r1["question_index"] == 2
    assert (r1["tools"], r1["condition"], r1["subq_similarity"], r1["correct"]) == (
        True,
        "mixed+tools",
        80.0,
        False,
    )
    assert (r2["verdict"], r2["correct"], r2["subq_similarity"]) == (
        "PROBLEM",
        None,
        None,
    )


def test_make_record_derived_fields():
    r = make_record(
        model="m",
        strategy="decomp",
        tools=True,
        file="f",
        question="q",
        judge_raw="Yes",
        subq_similarity_raw="70%",
    )
    assert (r["condition"], r["verdict"], r["correct"], r["subq_similarity"]) == (
        "decomp+tools",
        "YES",
        True,
        70.0,
    )
    assert r["schema_version"] == 1 and "netexplainer_version" in r


# --- runner -----------------------------------------------------------------


def test_conditions():
    assert conditions(["decomp", "decomp_tools"], [False]) == [
        ("decomp", False),
        ("decomp", True),
    ]
    assert conditions(["direct"], [False, True]) == [
        ("direct", False),
        ("direct", True),
    ]


class FakeDataset:
    def __init__(self, path, questions_path, context):
        self.processed_file = path.replace(".pcap", ".txt")
        with open(self.processed_file, "w") as f:
            f.write("No.|Time\n1 | 0.0\n")
        self.questions_subquestions = {q: ["ref sub"] for q in QUESTIONS}
        self.questions_answers = {q: 1 for q in QUESTIONS}


def _llm_responder(messages):
    text = messages[0].content
    if "generates multiple sub-questions" in text:
        return "sub 1\nsub 2"
    return "answer"


@pytest.fixture
def experiment(tmp_path, monkeypatch, fake_chat_model):
    data = tmp_path / "cleaned"
    data.mkdir()
    for name in ("b.pcap", "a.pcap", "ignored.txt"):
        (data / name).write_text("")
    monkeypatch.setattr("netexplainer.evaluation.runner.Dataset", FakeDataset)
    judge_calls = {"n": 0}

    def judge_responder(messages):
        judge_calls["n"] += 1
        if judge_calls["n"] == 1:
            raise RuntimeError("API down")
        return "80" if "percentage of similarity" in messages[0].content else "YES"

    judge = Judge(fake_chat_model(judge_responder), name="judge", delay=0)
    factory_calls = []

    def factory(backend, name, params, **kw):
        factory_calls.append((backend, name, params, kw))
        return fake_chat_model(_llm_responder)

    cfg = RunConfig(
        models=["phi4"],
        strategies=["mixed"],
        data_dir=data,
        questions_path=tmp_path / "q.yaml",
        output=tmp_path / "out" / "results.jsonl",
        tools=[False, True],
        direct_questions=(1,),
        max_files=1,
    )
    return SimpleNamespace(cfg=cfg, judge=judge, factory=factory, calls=factory_calls)


def test_run_experiment_writes_jsonl_and_resumes(experiment):
    e = experiment
    n = run_experiment(
        e.cfg, e.judge, chat_model_factory=e.factory, sleep=lambda s: None
    )
    assert n == 2 * len(QUESTIONS)  # 1 file x (no tools, tools) x questions
    backend, name, params, _ = e.calls[0]
    assert (backend, name, params.num_ctx, params.temperature) == (
        "ollama",
        "phi4:14b-fp16",
        16000,
        None,
    )
    records = list(read_jsonl(e.cfg.output))
    assert {r["file"] for r in records} == {"a.pcap"}  # sorted, max_files=1
    first = records[0]
    assert first["attempts"] == 2 and len(first["errors"]) == 1  # judge failed once
    assert first["decomposed"] is False and first["subq_similarity"] is None
    second = records[1]
    assert second["decomposed"] and second["subquestions"] == ["sub 1", "sub 2"]
    assert second["intermediate_answers"] == ["answer", "answer"]
    assert second["subq_similarity"] == 80.0 and second["correct"] is True
    assert second["llm_calls"] == 4 and second["ground_truth"] == "1"
    assert {r["condition"] for r in records} == {"mixed", "mixed+tools"}
    json.dumps(records)  # serializable

    with pytest.raises(FileExistsError):
        run_experiment(
            e.cfg, e.judge, chat_model_factory=e.factory, sleep=lambda s: None
        )
    e.cfg.resume = True
    assert (
        run_experiment(
            e.cfg, e.judge, chat_model_factory=e.factory, sleep=lambda s: None
        )
        == 0
    )
    assert len(completed_keys(e.cfg.output)) == 2 * len(QUESTIONS)


# --- cli --------------------------------------------------------------------


def test_cli_convert_and_analyze(tmp_path, capsys):
    log = tmp_path / "old.log"
    log.write_text(LEGACY_LOG)
    questions = tmp_path / "q.yaml"
    questions.write_text(
        "questions:\n"
        + "".join(f'  - question: "{q}"\n    subquestions: []\n' for q in QUESTIONS)
    )
    out = tmp_path / "legacy.jsonl"
    common = ["--log-file", str(tmp_path / "x.log")]
    assert (
        main(
            common
            + ["convert-log", str(log), "-o", str(out), "--questions", str(questions)]
        )
        == 0
    )
    assert len(list(read_jsonl(out))) == 3
    capsys.readouterr()
    assert main(common + ["analyze", str(out), "--by-question"]) == 0
    text = capsys.readouterr().out
    assert "## accuracy" in text and "## mcnemar" in text and "decomp+tools" in text


def test_cli_run_builds_config(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("netexplainer.llm.build_chat_model", lambda *a, **k: object())
    monkeypatch.setattr(
        "netexplainer.evaluation.runner.run_experiment",
        lambda cfg, judge: seen.update(cfg=cfg, judge=judge) or 0,
    )
    argv = [
        "--log-file",
        str(tmp_path / "x.log"),
        "run",
        "--models",
        "phi4",
        "my-model",
        "--backend",
        "openai",
        "--base-url",
        "http://localhost:8000/v1",
        "--strategy",
        "decomp",
        "mixed",
        "--direct-questions",
        "1,4",
        "--tool-ablation",
        "--temperature",
        "0",
        "--seed",
        "7",
        "--max-files",
        "3",
        "--output-dir",
        str(tmp_path / "res"),
        "--resume",
    ]
    assert main(argv) == 0
    cfg = seen["cfg"]
    assert cfg.models == ["phi4", "my-model"] and cfg.backend == "openai"
    assert cfg.strategies == ["decomp", "mixed"] and cfg.direct_questions == (1, 4)
    assert cfg.tools == [False, True] and cfg.resume and cfg.max_files == 3
    assert (cfg.params.temperature, cfg.params.seed, cfg.params.top_p) == (0.0, 7, None)
    assert cfg.output == tmp_path / "res" / "results.jsonl"
    assert seen["judge"].name == "gemma-3-27b-it"


def test_plots(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    from unittest.mock import patch

    from netexplainer.evaluation.plots import generate_plots

    records = list(parse_legacy_log(LEGACY_LOG.splitlines(), "decomp", QUESTIONS))
    with patch("plotly.graph_objects.Figure.write_image"):
        generate_plots(records, tmp_path)
    assert (tmp_path / "phi4_decomp" / "answers.txt").exists()
    assert (tmp_path / "phi4_decomp_tools" / "subquestions_similarity.txt").exists()
