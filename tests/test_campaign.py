"""Runner features for the revision: tool modes, react, question sets, scale."""

from scapy.all import IP, UDP, Ether, wrpcap

from netexplainer.cli import build_parser, main
from netexplainer.cli_run import build_judge as _build_judge
from netexplainer.evaluation.judge import Judge
from netexplainer.evaluation.records import completed_keys, read_jsonl
from netexplainer.evaluation.runner import RunConfig, list_traces, run_experiment

QUESTIONS = ["Q ten?", "Q eleven?"]


class FakeDataset:
    seen = []

    def __init__(self, path, questions_path, context, question_set="core"):
        FakeDataset.seen.append(question_set)
        self.processed_file = path.replace(".pcap", ".txt")
        with open(self.processed_file, "w") as f:
            f.write("No.|Time\n")
        self.questions_subquestions = {q: ["ref"] for q in QUESTIONS}
        self.questions_answers = {q: 1 for q in QUESTIONS}
        self.question_indexes = {QUESTIONS[0]: 10, QUESTIONS[1]: 11}
        self.n_packets = 7


def _run(tmp_path, monkeypatch, fake_chat_model, **cfg_kw):
    data = tmp_path / "cleaned"
    data.mkdir(exist_ok=True)
    (data / "a.pcap").write_text("")
    monkeypatch.setattr("netexplainer.evaluation.runner.Dataset", FakeDataset)

    def react_llm(messages):
        text = messages[0].content
        return (
            "Final Answer: 6"
            if "Observation: " in text.split("Trace:")[-1]
            else "Action: calculator[2*3]"
        )

    judge = Judge(fake_chat_model(lambda m: "YES"), name="j", delay=0)
    cfg = RunConfig(
        models=["m"],
        data_dir=data,
        questions_path=tmp_path / "q.yaml",
        output=tmp_path / "out" / "results.jsonl",
        **cfg_kw,
    )
    n = run_experiment(
        cfg,
        judge,
        chat_model_factory=lambda *a, **k: fake_chat_model(react_llm),
        sleep=lambda s: None,
    )
    return cfg, n


def test_react_and_tool_modes_are_recorded(tmp_path, monkeypatch, fake_chat_model):
    cfg, n = _run(
        tmp_path,
        monkeypatch,
        fake_chat_model,
        strategies=["react"],
        tools=["none", "callable", "declared"],
        question_set="extended",
    )
    assert n == 3 * 2 and FakeDataset.seen[-1] == "extended"
    records = list(read_jsonl(cfg.output))
    by_mode = {r["tool_mode"]: r for r in records if r["question_index"] == 10}
    assert set(by_mode) == {"none", "callable", "declared"}
    assert (
        by_mode["callable"]["tools"] is True and by_mode["declared"]["tools"] is False
    )
    assert by_mode["declared"]["condition"] == "react+declared"
    assert by_mode["callable"]["condition"] == "react+tools"
    # none: the model's action is rejected, uncounted; declared: counted+blocked
    assert (by_mode["none"]["tool_calls"], by_mode["none"]["tool_calls_blocked"]) == (
        0,
        0,
    )
    assert (
        by_mode["declared"]["tool_calls"],
        by_mode["declared"]["tool_calls_blocked"],
    ) == (1, 1)
    assert by_mode["callable"]["tool_calls_blocked"] == 0
    assert by_mode["callable"]["react_steps"][1] == "Observation: 6"
    assert (
        by_mode["callable"]["n_packets"] == 7 and by_mode["callable"]["llm_calls"] == 2
    )
    assert len(completed_keys(cfg.output)) == 6  # modes are distinct resume keys


def test_cli_tool_modes_and_options(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "netexplainer.evaluation.runner.run_experiment",
        lambda cfg, judge: seen.update(cfg=cfg) or 0,
    )
    base = ["--log-file", str(tmp_path / "x.log"), "run", "--models", "m"]
    extra = ["--judge-backend", "none"]
    args = ["--strategy", "react", "cot", "--tools-declared-not-callable"]
    assert (
        main(base + extra + args + ["--question-set", "all", "--max-packets", "512"])
        == 0
    )
    cfg = seen["cfg"]
    assert cfg.tools == ["declared"] and cfg.question_set == "all"
    assert cfg.strategies == ["react", "cot"] and cfg.max_packets == 512
    assert main(base + extra + ["--tools", "--react-max-steps", "3"]) == 0
    assert seen["cfg"].tools == ["callable"] and seen["cfg"].react_max_steps == 3
    assert seen["cfg"].question_set == "core"


def test_ollama_judge_is_built_with_url_and_context():
    args = build_parser().parse_args(
        [
            "run",
            "--judge-backend",
            "ollama",
            "--judge-model",
            "gemma3:27b",
            "--judge-base-url",
            "http://localhost:11438",
            "--judge-num-ctx",
            "8192",
        ]
    )
    judge = _build_judge(args)
    llm = judge.llm
    assert (judge.name, llm.model, llm.base_url, llm.num_ctx, llm.temperature) == (
        "gemma3:27b",
        "gemma3:27b",
        "http://localhost:11438",
        8192,
        0.0,
    )


def test_list_traces_by_packet_count(tmp_path):
    for name, n in (("small.pcap", 3), ("mid.pcap", 6), ("big.pcap", 10)):
        pkt = Ether() / IP(dst="1.1.1.1") / UDP()
        wrpcap(str(tmp_path / name), [pkt] * n)
    names = lambda **kw: [p.name for p in list_traces(tmp_path, **kw)]  # noqa: E731
    assert names() == ["big.pcap", "mid.pcap", "small.pcap"]
    assert names(max_packets=6) == ["mid.pcap", "small.pcap"]
    assert names(min_packets=4, max_packets=6) == ["mid.pcap"]
    assert names(max_packets=6, max_files=1) == ["mid.pcap"]


def test_clean_data_accepts_directories():
    args = build_parser().parse_args(
        ["clean-data", "512", "--raw-dir", "r", "--cleaned-dir", "c512"]
    )
    assert (args.max_packets, str(args.raw_dir), str(args.cleaned_dir)) == (
        512,
        "r",
        "c512",
    )
