"""Tests for Converse responses that carry redacted reasoning.

ReasoningContentBlock is a union: a block holds either reasoningText or redactedContent.
"""

from unittest import mock

import pytest

# api.models.bedrock creates boto3 clients, lists models and loads the tiktoken
# encoding when imported. Stub those so the import makes no network calls.
with mock.patch("boto3.client"), mock.patch("tiktoken.get_encoding"):
    from api.models import bedrock as bedrock_module  # noqa: F401

from api.models.bedrock import BedrockModel

REDACTED = {"reasoningContent": {"redactedContent": b"opaque"}}
REASONING = {"reasoningContent": {"reasoningText": {"text": "Think.", "signature": "sig"}}}


@pytest.fixture
def model():
    return BedrockModel()


def _content(model, content):
    response = model._create_response(model="m", message_id="chatcmpl-1", content=content, finish_reason="end_turn")
    return response.choices[0].message.content


def test_redacted_reasoning_is_skipped(model):
    assert _content(model, [REDACTED, {"text": "Hello."}]) == "Hello."


def test_redacted_block_keeps_earlier_reasoning(model):
    assert _content(model, [REASONING, REDACTED, {"text": "Hello."}]) == "<think>Think.</think>Hello."


def test_redacted_reasoning_counts_no_reasoning_tokens(model):
    assert model._estimate_reasoning_tokens([REDACTED, {"text": "Hello."}]) == 0
