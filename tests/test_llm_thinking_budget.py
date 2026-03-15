from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from workflow_compiler.core.llm import client
from workflow_compiler.core.llm.config import (
    MODEL_CONFIG_JSON_ENV,
    serialize_model_config_payload,
    set_default_model_config_payload,
)
from workflow_compiler.core.llm.thinking_budget import (
    DEFAULT_THINKING_BUDGET_CUTOFF_TEXT,
)


class _DummyUsage(SimpleNamespace):
    pass


class _DummyChatEndpoint:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _DummyCompletionsEndpoint:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected completions.create() call")
        return self.responses.pop(0)


class _DummyOpenAIClient:
    def __init__(self, chat_response=None, chat_error: Exception | None = None, completion_responses=None, *args, **kwargs):
        self.chat = SimpleNamespace(
            completions=_DummyChatEndpoint(response=chat_response, error=chat_error)
        )
        self.completions = _DummyCompletionsEndpoint(completion_responses)


def _build_cfg(**overrides) -> client.LLMConfig:
    raw = {
        "name": "qwen3-4b",
        "api_type": "openai",
        "model": "qwen3-4b",
        "temperature": 1,
        "top_p": 1,
        "api_key": "dummy",
        "base_url": "http://127.0.0.1:4000",
        "hf_model_name": "Qwen/Qwen3-4B",
        "enable_thinking_budget": True,
    }
    raw.update(overrides)
    return client.LLMConfig(raw)


def test_integer_budget_uses_single_chat_call(monkeypatch):
    response = SimpleNamespace(
        usage=_DummyUsage(prompt_tokens=11, completion_tokens=7),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="<think>hidden</think>\n\nfinal answer"
                )
            )
        ],
    )

    dummy_clients = []

    def factory(*args, **kwargs):
        inst = _DummyOpenAIClient(chat_response=response, *args, **kwargs)
        dummy_clients.append(inst)
        return inst

    monkeypatch.setattr(client, "AsyncOpenAI", factory)

    llm = client.AsyncLLM(_build_cfg())
    result = asyncio.run(llm.call_with_thinking_budget("Solve", 17, return_io_tokens=True))

    assert result == ("final answer", 11, 7)
    chat_calls = dummy_clients[0].chat.completions.calls
    assert len(chat_calls) == 1
    assert dummy_clients[0].completions.calls == []
    extra_body = chat_calls[0]["extra_body"]
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert extra_body["vllm_xargs"]["thinking_budget"] == 17
    assert extra_body["vllm_xargs"]["thinking_cutoff_text"] == DEFAULT_THINKING_BUDGET_CUTOFF_TEXT
    assert llm.get_usage_summary()["call_count"] == 1


def test_integer_budget_vllm_backend_surfaces_passthrough_errors(monkeypatch):
    def factory(*args, **kwargs):
        return _DummyOpenAIClient(chat_error=ValueError("proxy rejected field"), *args, **kwargs)

    monkeypatch.setattr(client, "AsyncOpenAI", factory)

    llm = client.AsyncLLM(_build_cfg())
    with pytest.raises(RuntimeError, match="LiteLLM preserves extra_body.vllm_xargs"):
        asyncio.run(llm.call_with_thinking_budget("Solve", 5))


def test_create_llm_instance_uses_profile_endpoint_role(monkeypatch, tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "endpoints:",
                '  local_base_url: "http://127.0.0.1:4000"',
                '  profile_base_url: "http://profile-host:4000"',
                "models:",
                "  qwen35-9b-awq:",
                '    api_type: "openai"',
                '    api_key: "dummy"',
                '    hf_model_name: "QuantTrio/Qwen3.5-9B-AWQ"',
                "    enable_thinking_budget: true",
            ]
        ),
        encoding="utf-8",
    )

    created = []

    def factory(*args, **kwargs):
        created.append(kwargs)
        return _DummyOpenAIClient(*args, **kwargs)

    monkeypatch.setenv("WORKFLOW_COMPILER_CONFIG", str(config_path))
    monkeypatch.setattr(client, "AsyncOpenAI", factory)
    client.LLMsConfig._default_config = None

    try:
        llm = client.create_llm_instance("qwen35-9b-awq", endpoint_role="profile")
    finally:
        client.LLMsConfig._default_config = None

    assert llm._request_model == "qwen35-9b-awq"
    assert created[0]["base_url"] == "http://profile-host:4000"


def test_default_endpoint_role_falls_back_to_local_base_url(monkeypatch, tmp_path):
    payload = {
        "endpoints": {
            "local_base_url": "http://127.0.0.1:4000",
            "profile_base_url": "http://profile-host:4000",
        },
        "models": {
            "qwen35-4b": {
                "api_type": "openai",
                "api_key": "dummy",
                "hf_model_name": "Qwen/Qwen3.5-4B",
                "enable_thinking_budget": True,
            }
        },
    }

    monkeypatch.setenv(MODEL_CONFIG_JSON_ENV, serialize_model_config_payload(payload))
    set_default_model_config_payload(payload)
    client.LLMsConfig._default_config = None

    try:
        cfg = client.LLMsConfig.default().get("qwen35-4b")
    finally:
        set_default_model_config_payload(None)
        client.LLMsConfig._default_config = None

    assert cfg.base_url == "http://127.0.0.1:4000"
