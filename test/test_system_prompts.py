"""Tests for how system and developer messages become Converse system blocks."""

from unittest import mock

import pytest

# api.models.bedrock creates boto3 clients, lists models and loads the tiktoken
# encoding when imported. Stub those so the import makes no network calls.
with mock.patch("boto3.client"), mock.patch("tiktoken.get_encoding"):
    from api.models import bedrock as bedrock_module  # noqa: F401

from api.models.bedrock import BedrockModel
from api.schema import ChatRequest

MODEL_ID = "amazon.titan-text-express-v1"  # no prompt caching, so no cachePoint is added


@pytest.fixture
def model():
    return BedrockModel()


def _system(model, *messages):
    request = ChatRequest.model_validate(
        {"model": MODEL_ID, "messages": [*messages, {"role": "user", "content": "Hi"}]}
    )
    return model._parse_system_prompts(request)


def test_string_content_is_one_block(model):
    assert _system(model, {"role": "system", "content": "Be brief."}) == [{"text": "Be brief."}]


@pytest.mark.parametrize("role", ["system", "developer"])
def test_text_parts_become_one_block_each(model, role):
    message = {"role": role, "content": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "Be kind."}]}
    assert _system(model, message) == [{"text": "Be brief."}, {"text": "Be kind."}]
