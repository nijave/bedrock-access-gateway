"""Tests for how a Converse response becomes a non-streamed chat completion."""

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


def _content(model, content):
    response = model._create_response(model="m", message_id="chatcmpl-1", content=content, finish_reason="end_turn")
    return response.choices[0].message.content


def test_all_text_blocks_are_kept(model):
    assert _content(model, [{"text": "Hello, "}, {"text": "world."}]) == "Hello, world."


def test_text_blocks_follow_reasoning(model):
    content = [
        {"reasoningContent": {"reasoningText": {"text": "Think."}}},
        {"text": "Hello, "},
        {"text": "world."},
    ]
    assert _content(model, content) == "<think>Think.</think>Hello, world."
