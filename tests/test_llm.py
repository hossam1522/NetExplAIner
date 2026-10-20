import hashlib
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from netexplainer.llm import (
    GenerationParams,
    LLMClient,
    Pipeline,
    build_chat_model,
    calculator,
    prompts,
    resolve_model,
)
from netexplainer.llm.pipeline import uses_decomposition

# sha256 of the prompts used in the paper. If one of these fails, a prompt
# changed and results are no longer comparable with the paper.
PAPER_PROMPT_HASHES = {
    "DECOMPOSITION": "d575ad8517dd0caf2cf953f097cc6c1b7aa09ca38c2364fafe8579d9ebf57a6f",
    "ANSWER": "cd93b8f32a65433e5c9c78b8b9eb1b5025ca0dfa32bea52fc2e765e296917902",
    "AGGREGATION": "f25bfe12a6bf49cf34ff6c21e0201cfa3203f82b2ff94f0aa5a97b5a87ccfc31",
    "JUDGE_SUBQUESTIONS": "f37278fe702e50efd0a6ac997c5e7dfe042d2fd757d0511ee28486a9da661a59",
    "JUDGE_ANSWER": "c9eaebbd0904b3634b395d1703795b24e8b729a6d421f46c16cdd1e4118d87ae",
}


@pytest.mark.parametrize("name,digest", PAPER_PROMPT_HASHES.items())
def test_paper_prompts_unchanged(name, digest):
    text = getattr(prompts, name)
    assert hashlib.sha256(text.encode()).hexdigest() == digest


def test_format_qa_pairs():
    text = prompts.format_qa_pairs(["q1", "q2"], ["a1", "a2"])
    assert text == "Question 1: q1\nAnswer 1: a1\n\nQuestion 2: q2\nAnswer 2: a2"


def test_calculator():
    assert calculator.invoke("2 + 3 * 4") == "14"
    assert float(calculator.invoke("pi * 2")) == pytest.approx(6.283185, abs=1e-5)


# --- backends ---------------------------------------------------------------


@patch("langchain_ollama.ChatOllama")
def test_ollama_registry_defaults_match_paper(mock):
    spec = resolve_model("mistral-7b")
    build_chat_model(spec.backend, spec.name, spec.params, extra=spec.extra)
    # No temperature/top_p/top_k: Ollama defaults, as in the paper.
    mock.assert_called_once_with(model="mistral", num_ctx=32768)


@patch("langchain_ollama.ChatOllama")
def test_ollama_explicit_params(mock):
    params = GenerationParams(num_ctx=4096).override(
        GenerationParams(temperature=0.0, top_p=0.5, top_k=10, seed=1)
    )
    build_chat_model("ollama", "m", params, base_url="http://h:11434")
    mock.assert_called_once_with(
        model="m",
        num_ctx=4096,
        temperature=0.0,
        top_p=0.5,
        top_k=10,
        seed=1,
        base_url="http://h:11434",
    )


@patch("langchain_ollama.ChatOllama")
def test_cogito_reasoning_extra(mock):
    spec = resolve_model("cogito-8b")
    build_chat_model(spec.backend, spec.name, spec.params, extra=spec.extra)
    mock.assert_called_once_with(model="cogito:8b", num_ctx=32000, reasoning=True)


@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_google_registry_defaults_match_paper(mock):
    spec = resolve_model("gemini-2.0-flash")
    build_chat_model(spec.backend, spec.name, spec.params)
    mock.assert_called_once_with(
        model="gemini-2.0-flash", temperature=0, max_tokens=None, timeout=None
    )


@patch("langchain_openai.ChatOpenAI")
def test_openai_compatible(mock, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    build_chat_model(
        "openai",
        "Qwen/Qwen2.5-32B-Instruct",
        GenerationParams(temperature=0.2, top_k=20, seed=3),
        base_url="http://localhost:8000/v1",
    )
    mock.assert_called_once_with(
        model="Qwen/Qwen2.5-32B-Instruct",
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        temperature=0.2,
        seed=3,
        extra_body={"top_k": 20},
    )


def test_unknown_backend():
    with pytest.raises(ValueError):
        build_chat_model("nope", "m")


def test_resolve_unregistered_model_uses_cli_backend():
    spec = resolve_model("my-model", default_backend="openai")
    assert (spec.name, spec.backend, spec.context) == ("my-model", "openai", "big")


# --- client -----------------------------------------------------------------


def test_client_tool_loop_and_usage(fake_chat_model):
    replies = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculator", "args": {"expression": "6*7"}, "id": "c1"}
                ],
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
            ),
            AIMessage(
                content="42",
                usage_metadata={
                    "input_tokens": 15,
                    "output_tokens": 1,
                    "total_tokens": 16,
                },
            ),
        ]
    )
    model = fake_chat_model(lambda m: next(replies))
    client = LLMClient(model, tools=True)
    assert model.bound_tools == [calculator]
    assert client.invoke([]) == "42"
    # second call received the tool result
    assert model.calls[1][-1].content == "42"
    usage = client.reset_usage()
    assert (usage.llm_calls, usage.tokens_in, usage.tokens_out, usage.tool_calls) == (
        2,
        25,
        3,
        1,
    )
    assert client.usage.llm_calls == 0


# --- pipeline ---------------------------------------------------------------


def _responder(messages):
    text = messages[0].content
    if "generates multiple sub-questions" in text:
        return "1. sub a\n\n2. sub b\n"
    if "Q+A pairs" in text:
        return "final"
    if "Let's think step by step" in text:
        return "cot answer"
    return "partial"


def _pipeline(fake_chat_model, **kw):
    model = fake_chat_model(_responder)
    return Pipeline(LLMClient(model), "TRACE", **kw), model


def test_decomposition_pipeline(fake_chat_model):
    pipeline, model = _pipeline(fake_chat_model)
    result = pipeline.run("decomp", "Q?", 1)
    assert result.decomposed
    assert result.subquestions == ["1. sub a", "2. sub b"]
    assert result.intermediate_answers == ["partial", "partial"]
    assert result.final_answer == "final"
    assert len(model.calls) == 4
    # answers see the trace, aggregation g only sees the Q+A pairs
    assert "TRACE" in model.calls[1][0].content
    agg = model.calls[3][0].content
    assert "TRACE" not in agg and "Question 1: 1. sub a\nAnswer 1: partial" in agg


@pytest.mark.parametrize(
    "strategy,index,decomposed",
    [
        ("direct", 3, False),
        ("mixed", 1, False),
        ("mixed", 3, True),
        ("decomp", 1, True),
        ("decomp_tools", 1, True),
        ("cot", 1, False),
    ],
)
def test_strategy_routing(fake_chat_model, strategy, index, decomposed):
    pipeline, _ = _pipeline(fake_chat_model)
    result = pipeline.run(strategy, "Q?", index)
    assert result.decomposed is decomposed
    assert uses_decomposition(strategy, index) is decomposed
    if strategy == "cot":
        assert result.final_answer == "cot answer"


def test_mixed_custom_direct_questions(fake_chat_model):
    pipeline, _ = _pipeline(fake_chat_model)
    assert not pipeline.run("mixed", "Q?", 3, direct_questions=[3]).decomposed


def test_request_delay(fake_chat_model):
    sleeps = []
    pipeline, _ = _pipeline(fake_chat_model, request_delay=2.5, sleep=sleeps.append)
    pipeline.run("decomp", "Q?", 1)
    assert sleeps == [2.5, 2.5, 2.5]  # 2 answers + aggregation, not decomposition
