"""Tests for how ConverseStream events become streamed chat completion chunks."""

from unittest import mock

import pytest

# api.models.bedrock creates boto3 clients, lists models and loads the tiktoken
# encoding when imported. Stub those so the import makes no network calls.
with mock.patch("boto3.client"), mock.patch("tiktoken.get_encoding"):
    from api.models import bedrock as bedrock_module  # noqa: F401

from api.models.bedrock import BedrockModel


@pytest.fixture
def model():
    return BedrockModel()


def _stream(model, chunks, model_id="global.moonshotai.kimi-k3"):
    # chat_stream resets this state before each stream.
    model.think_emitted = False
    model._tool_call_index_map = {}
    model._next_tool_index = 0
    model._tool_call_has_args = {}
    responses = [model._create_response_stream(model_id, "chatcmpl-1", chunk) for chunk in chunks]
    return [r for r in responses if r]


def test_unsigned_reasoning_followed_by_a_tool_call_keeps_the_finish_reason(model):
    # Kimi and GLM go straight from unsigned reasoning to a tool call, with no text block.
    responses = _stream(
        model,
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Look it up."}}}},
            {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "c1", "name": "f"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{}"}}}},
            {"messageStop": {"stopReason": "tool_use"}},
        ],
    )

    deltas = [r.choices[0].delta for r in responses]
    assert deltas[1].content == "</think>"
    assert deltas[1].tool_calls[0].id == "c1"
    assert "".join(d.content or "" for d in deltas) == "<think>Look it up.</think>"
    assert responses[-1].choices[0].finish_reason == "tool_calls"


def test_reasoning_left_open_at_message_stop_is_closed_with_the_finish_reason(model):
    responses = _stream(
        model,
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Hmm."}}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ],
    )

    last = responses[-1].choices[0]
    assert last.delta.content == "</think>"
    assert last.finish_reason == "stop"
    assert model.think_emitted is False


def _tool_call_deltas(responses):
    return [tc for r in responses for tc in (r.choices[0].delta.tool_calls or [])]


def test_tool_call_as_the_first_block_gets_index_zero(model):
    responses = _stream(
        model,
        [
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "c1", "name": "f"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"a": 1}'}}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
        ],
    )

    assert {tc.index for tc in _tool_call_deltas(responses)} == {0}


def test_tool_call_indexes_count_only_tool_blocks(model):
    responses = _stream(
        model,
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Checking."}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "c1", "name": "f"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{}"}}}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"contentBlockStart": {"contentBlockIndex": 2, "start": {"toolUse": {"toolUseId": "c2", "name": "g"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {"toolUse": {"input": "{}"}}}},
            {"contentBlockStop": {"contentBlockIndex": 2}},
            {"messageStop": {"stopReason": "tool_use"}},
        ],
        model_id="anthropic.claude-sonnet-4-5-20250929-v1:0",
    )

    starts = [tc for tc in _tool_call_deltas(responses) if tc.id]
    assert [(tc.id, tc.index) for tc in starts] == [("c1", 0), ("c2", 1)]


def test_empty_argument_deltas_are_dropped_so_arguments_stay_valid_json(model):
    responses = _stream(
        model,
        [
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "c1", "name": "f"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": ""}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"a": '}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": "1}"}}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
        ],
    )

    arguments = "".join(tc.function.arguments or "" for tc in _tool_call_deltas(responses))
    assert arguments == '{"a": 1}'


def test_tool_call_without_arguments_gets_an_empty_object(model):
    responses = _stream(
        model,
        [
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "c1", "name": "f"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": ""}}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
        ],
    )

    arguments = "".join(tc.function.arguments or "" for tc in _tool_call_deltas(responses))
    assert arguments == "{}"
