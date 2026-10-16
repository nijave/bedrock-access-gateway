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
