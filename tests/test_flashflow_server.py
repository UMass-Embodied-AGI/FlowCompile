from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from flashflow.backends.base import BaseBackend
from flashflow.runtime import FlashFlowRuntime
from flashflow.server import create_app
from flashflow.types import AliasInfo, BackendResult


class _DummyVLLMBackend(BaseBackend):
    def __init__(self, model_name, metadata):
        super().__init__(model_name, metadata)
        self.initialized = False
        self.sleep_calls = 0
        self.sleep_levels = []
        self.wake_calls = 0

    async def initialize(self) -> None:
        self.initialized = True

    async def warmup(self) -> None:
        return None

    async def wake(self) -> None:
        self.wake_calls += 1

    async def sleep(self, level=1) -> None:
        self.sleep_calls += 1
        self.sleep_levels.append(level)

    async def generate_chat(self, messages, alias_info: AliasInfo, request):
        return await self.generate_completion(messages[-1]["content"], alias_info, request)

    async def generate_completion(self, prompt: str, alias_info: AliasInfo, request):
        if alias_info.budget == "unlimited":
            return BackendResult(
                text="unlimited answer",
                input_tokens=4,
                output_tokens=6,
                model_name=self.model_name,
            )
        return BackendResult(
            text="two-stage answer",
            input_tokens=11,
            output_tokens=7,
            model_name=self.model_name,
        )


class _DummyAzureBackend(BaseBackend):
    async def initialize(self) -> None:
        return None

    async def generate_chat(self, messages, alias_info: AliasInfo, request):
        return await self.generate_completion(messages[-1]["content"], alias_info, request)

    async def generate_completion(self, prompt: str, alias_info: AliasInfo, request):
        if alias_info.budget not in (None, "unlimited"):
            raise ValueError(
                f"Azure-backed model '{alias_info.base_model}' only supports unlimited thinking mode in FlashFlow."
            )
        return BackendResult(
            text="azure answer",
            input_tokens=5,
            output_tokens=3,
            model_name=self.model_name,
        )


def _write_exported_dag(tmp_path: Path) -> Path:
    payload = {
        "version": "v1",
        "name": "flashflow_test",
        "metadata": {
            "flashflow": {
                "schema_version": "flashflow.export.v1",
                "aliases": {
                    "qwen35-4b_budget_2000": {
                        "model_alias": "qwen35-4b_budget_2000",
                        "base_model": "qwen35-4b",
                        "budget": 2000,
                        "backend": "vllm",
                        "default_thinking_strategy": "two_stage",
                    },
                    "qwen35-4b_budget_unlimited": {
                        "model_alias": "qwen35-4b_budget_unlimited",
                        "base_model": "qwen35-4b",
                        "budget": "unlimited",
                        "backend": "vllm",
                        "default_thinking_strategy": "vllm_plugin",
                    },
                    "gpt-5-mini_budget_unlimited": {
                        "model_alias": "gpt-5-mini_budget_unlimited",
                        "base_model": "gpt-5-mini",
                        "budget": "unlimited",
                        "backend": "azure",
                        "default_thinking_strategy": "unlimited_only",
                    },
                    "gpt-5-mini_budget_100": {
                        "model_alias": "gpt-5-mini_budget_100",
                        "base_model": "gpt-5-mini",
                        "budget": 100,
                        "backend": "azure",
                        "default_thinking_strategy": "unlimited_only",
                    },
                },
                "models": {
                    "qwen35-4b": {
                        "backend": "vllm",
                        "artifact_id": "Qwen/Qwen3.5-4B",
                        "tokenizer": "Qwen/Qwen3.5-4B",
                    },
                    "gpt-5-mini": {
                        "backend": "azure",
                        "azure_endpoint": "https://example.openai.azure.com",
                        "azure_deployment": "gpt-5-mini-2024-07-18",
                        "api_key": "secret",
                        "api_version": "2024-10-21",
                    },
                },
            }
        },
        "nodes": [],
        "edges": [],
    }
    dag_file = tmp_path / "workflow_dag.json"
    dag_file.write_text(json.dumps(payload), encoding="utf-8")
    return dag_file


def _patch_build_backend(monkeypatch):
    def _build_backend(self, model_name, meta):
        if meta.get("backend") == "azure":
            return _DummyAzureBackend(model_name, meta)
        return _DummyVLLMBackend(model_name, meta)

    monkeypatch.setattr(FlashFlowRuntime, "_build_backend", _build_backend)


def test_flashflow_server_routes_and_token_ledger(monkeypatch, tmp_path):
    _patch_build_backend(monkeypatch)
    runtime = FlashFlowRuntime(str(_write_exported_dag(tmp_path)))
    with TestClient(create_app(runtime)) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        listed = {item["id"] for item in models.json()["data"]}
        assert "qwen35-4b_budget_2000" in listed
        assert "gpt-5-mini_budget_unlimited" in listed

        vllm_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen35-4b_budget_2000",
                "messages": [{"role": "user", "content": "solve"}],
            },
        )
        assert vllm_resp.status_code == 200
        assert vllm_resp.json()["usage"] == {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }

        azure_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5-mini_budget_unlimited",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert azure_resp.status_code == 200

        usage = client.get("/v1/flashflow/token_usage")
        assert usage.status_code == 200
        assert usage.json()["data"] == {
            "qwen35-4b": {"input": 11, "output": 7},
            "gpt-5-mini": {"input": 5, "output": 3},
        }

        reset = client.post("/v1/flashflow/token_usage/reset")
        assert reset.status_code == 200
        assert client.get("/v1/flashflow/token_usage").json()["data"] == {}

    vllm_backend = runtime.backends["qwen35-4b"]
    assert vllm_backend.sleep_calls >= 1
    assert set(vllm_backend.sleep_levels) == {1}


def test_flashflow_rejects_integer_budget_for_azure(monkeypatch, tmp_path):
    _patch_build_backend(monkeypatch)
    runtime = FlashFlowRuntime(str(_write_exported_dag(tmp_path)))
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5-mini_budget_100",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 400
        assert "only supports unlimited thinking mode" in response.json()["detail"]
