from __future__ import annotations

from workflow_compiler.core.llm import client


class _DummyOpenAI:
    def __init__(self, *args, **kwargs):
        pass


def _build_cfg() -> client.LLMConfig:
    return client.LLMConfig(
        {
            "name": "qwen3-14b",
            "api_type": "openai",
            "model": "qwen3-14b",
            "temperature": 1,
            "top_p": 1,
            "api_key": "dummy",
            "base_url": "http://127.0.0.1:4000",
            "hf_model_name": "Qwen/Qwen3-14B",
            "enable_thinking_budget": True,
        }
    )


def test_tokenizer_loading_uses_cache_and_local_only_for_local_backend(monkeypatch):
    calls = []

    def fake_from_pretrained(model_name, **kwargs):
        calls.append((model_name, dict(kwargs)))
        return object()

    monkeypatch.delenv("FLOWCOMPILE_HF_LOCAL_FILES_ONLY", raising=False)
    monkeypatch.setattr(client, "AsyncOpenAI", _DummyOpenAI)
    monkeypatch.setattr(client.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    client.AsyncLLM._TOKENIZER_CACHE = {}

    cfg = _build_cfg()
    _ = client.AsyncLLM(cfg)
    _ = client.AsyncLLM(cfg)

    assert len(calls) == 1
    assert calls[0][0] == "Qwen/Qwen3-14B"
    assert calls[0][1].get("local_files_only") is True


def test_tokenizer_loading_env_override_can_disable_local_only(monkeypatch):
    calls = []

    def fake_from_pretrained(model_name, **kwargs):
        calls.append((model_name, dict(kwargs)))
        return object()

    monkeypatch.setenv("FLOWCOMPILE_HF_LOCAL_FILES_ONLY", "0")
    monkeypatch.setattr(client, "AsyncOpenAI", _DummyOpenAI)
    monkeypatch.setattr(client.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    client.AsyncLLM._TOKENIZER_CACHE = {}

    cfg = _build_cfg()
    _ = client.AsyncLLM(cfg)

    assert len(calls) == 1
    assert "local_files_only" not in calls[0][1]
