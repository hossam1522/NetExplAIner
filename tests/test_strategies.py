"""direct/cot/react strategies and the tool modes (none/callable/declared)."""

import pytest
from langchain_core.messages import AIMessage

from netexplainer.llm import LLMClient, Pipeline
from netexplainer.llm.client import MAX_TOOL_ROUNDS, TOOL_UNAVAILABLE


def _pipeline(fake_chat_model, replies, tools="none", **kw):
    it = iter(replies)
    model = fake_chat_model(lambda m: next(it))
    return Pipeline(LLMClient(model, tools=tools), "TRACE", **kw), model


def test_direct_and_cot_are_single_calls_without_decomposition(fake_chat_model):
    for strategy in ("direct", "cot"):
        pipeline, model = _pipeline(fake_chat_model, ["ans"])
        result = pipeline.run(strategy, "How many?", 1)
        assert (result.final_answer, result.decomposed) == ("ans", False)
        assert len(model.calls) == 1
        text = model.calls[0][0].content
        assert 'Question: "How many?"' in text and "TRACE" in text
        assert ("step by step" in text) is (strategy == "cot")
        assert "sub-questions" not in text


def test_react_calculator_loop(fake_chat_model):
    replies = [
        "Thought: need the product\nAction: calculator[6*7]\nObservation: made up",
        "Thought: done\nFinal Answer: 42",
    ]
    pipeline, model = _pipeline(fake_chat_model, replies, tools="callable")
    result = pipeline.run("react", "6 times 7?", 1)
    assert result.final_answer == "42" and not result.decomposed
    assert result.steps[1] == "Observation: 42"  # real result, not the invented one
    assert "Observation: 42" in model.calls[1][0].content
    assert "TRACE" in model.calls[1][0].content  # full trace on every step
    usage = pipeline.client.reset_usage()
    assert (usage.llm_calls, usage.tool_calls, usage.tool_calls_blocked) == (2, 1, 0)


def test_react_tool_error_goes_back_to_the_model(fake_chat_model):
    replies = ["Action: calculator[1 +]", "Final Answer: 3"]
    pipeline, model = _pipeline(fake_chat_model, replies, tools="callable")
    assert pipeline.run("react", "Q?", 1).final_answer == "3"
    assert "Observation: Error" in model.calls[1][0].content


def test_react_step_cap_forces_final_answer(fake_chat_model):
    loop = ["Action: calculator[1+1]"] * 3 + ["forced"]
    pipeline, model = _pipeline(fake_chat_model, loop, "callable", react_max_steps=3)
    result = pipeline.run("react", "Q?", 1)
    assert result.final_answer == "forced" and len(model.calls) == 4
    assert pipeline.client.usage.tool_calls == 3


def test_react_plain_reply_is_the_answer_and_no_tools_prompt(fake_chat_model):
    pipeline, model = _pipeline(fake_chat_model, ["just 5"])
    assert pipeline.run("react", "Q?", 1).final_answer == "just 5"
    assert "You have no tools" in model.calls[0][0].content


def test_react_without_tools_rejects_actions_uncounted(fake_chat_model):
    pipeline, _ = _pipeline(
        fake_chat_model, ["Action: calculator[1]", "Final Answer: 1"]
    )
    pipeline.run("react", "Q?", 1)
    assert pipeline.client.usage.tool_calls == 0


def test_declared_tool_is_bound_but_not_executable(fake_chat_model):
    call = {"name": "calculator", "args": {"expression": "6*7"}, "id": "c1"}
    replies = [AIMessage(content="", tool_calls=[call]), AIMessage(content="42?")]
    pipeline, model = _pipeline(fake_chat_model, replies, tools="declared")
    assert model.bound_tools  # definition is part of the request
    assert pipeline.run("direct", "Q?", 1).final_answer == "42?"
    assert model.calls[1][-1].content == TOOL_UNAVAILABLE
    usage = pipeline.client.reset_usage()
    assert (usage.tool_calls, usage.tool_calls_blocked) == (1, 1)


def test_declared_react_blocks_the_calculator(fake_chat_model):
    replies = ["Action: calculator[6*7]", "Final Answer: 42"]
    pipeline, model = _pipeline(fake_chat_model, replies, tools="declared")
    pipeline.run("react", "Q?", 1)
    assert f"Observation: {TOOL_UNAVAILABLE}" in model.calls[1][0].content
    assert "calculator[<expression>]" in model.calls[0][0].content  # declared
    assert pipeline.client.usage.tool_calls_blocked == 1


def test_tool_loop_is_bounded(fake_chat_model):
    call = {"name": "calculator", "args": {"expression": "1"}, "id": "c"}
    model = fake_chat_model(lambda m: AIMessage(content="x", tool_calls=[call]))
    client = LLMClient(model, tools="declared")
    client.invoke([])
    assert client.usage.llm_calls == MAX_TOOL_ROUNDS + 1


def test_invalid_tool_mode(fake_chat_model):
    with pytest.raises(ValueError):
        LLMClient(fake_chat_model(lambda m: ""), tools="maybe")
