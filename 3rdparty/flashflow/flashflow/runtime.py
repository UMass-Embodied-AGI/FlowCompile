"""FlashFlow runtime service."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from flashflow.accounting import TokenLedger
from flashflow.config import get_flashflow_metadata, load_workflow_dag
from flashflow.types import AliasInfo, BackendResult

if TYPE_CHECKING:
    from flashflow.backends.base import BaseBackend


class FlashFlowRuntime:
    def __init__(self, workflow_dag_file: str, *, vllm_args: Optional[Dict[str, Any]] = None) -> None:
        self.workflow_dag_file = workflow_dag_file
        self.vllm_args = dict(vllm_args or {})
        self.workflow_dag = load_workflow_dag(workflow_dag_file)
        self.flashflow_metadata = get_flashflow_metadata(self.workflow_dag)
        self.aliases: Dict[str, AliasInfo] = {
            str(alias): AliasInfo.from_dict(info)
            for alias, info in (self.flashflow_metadata.get("aliases") or {}).items()
        }
        self.models_meta: Dict[str, Dict[str, Any]] = {
            str(name): dict(meta)
            for name, meta in (self.flashflow_metadata.get("models") or {}).items()
        }
        self.backends: Dict[str, "BaseBackend"] = {}
        self.ledger = TokenLedger()
        self._switch_lock = asyncio.Lock()
        self._active_vllm_model: Optional[str] = None

    async def startup(self) -> None:
        for model_name, meta in self.models_meta.items():
            backend = self._build_backend(model_name, meta)
            self.backends[model_name] = backend
            await backend.initialize()
            if meta.get("backend") == "vllm":
                await backend.warmup()
                await backend.sleep(level=1)
        self._active_vllm_model = None

    async def shutdown(self) -> None:
        for backend in self.backends.values():
            try:
                await backend.sleep(level=1)
            except Exception:
                pass

    def _build_backend(self, model_name: str, meta: Dict[str, Any]) -> "BaseBackend":
        backend = str(meta.get("backend") or "vllm").lower()
        if backend == "azure":
            from flashflow.backends.azure import AzureOpenAIBackend

            return AzureOpenAIBackend(model_name, meta)
        from flashflow.backends.vllm import VLLMBackend

        return VLLMBackend(model_name, meta, self.vllm_args)

    async def _resolve_backend(self, alias: str) -> tuple[AliasInfo, BaseBackend]:
        alias_info = self.aliases.get(alias)
        if alias_info is None:
            raise ValueError(f"Unknown model alias '{alias}'.")
        backend = self.backends.get(alias_info.base_model)
        if backend is None:
            raise ValueError(f"Backend for model '{alias_info.base_model}' is not initialized.")
        return alias_info, backend

    async def _run_vllm_chat(
        self,
        alias_info: AliasInfo,
        backend: BaseBackend,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
    ) -> BackendResult:
        async with self._switch_lock:
            if self._active_vllm_model and self._active_vllm_model != alias_info.base_model:
                await self.backends[self._active_vllm_model].sleep(level=1)
            await backend.wake()
            self._active_vllm_model = alias_info.base_model
            return await backend.generate_chat(messages, alias_info, body)

    async def _run_vllm_completion(
        self,
        alias_info: AliasInfo,
        backend: BaseBackend,
        prompt: str,
        body: Dict[str, Any],
    ) -> BackendResult:
        async with self._switch_lock:
            if self._active_vllm_model and self._active_vllm_model != alias_info.base_model:
                await self.backends[self._active_vllm_model].sleep(level=1)
            await backend.wake()
            self._active_vllm_model = alias_info.base_model
            return await backend.generate_completion(prompt, alias_info, body)

    async def handle_chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        model_alias = str(body.get("model") or "")
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("chat.completions requires a messages list.")
        if body.get("stream"):
            raise ValueError("stream=true is not supported in FlashFlow v1.")
        if int(body.get("n", 1)) != 1:
            raise ValueError("FlashFlow v1 only supports n=1.")
        alias_info, backend = await self._resolve_backend(model_alias)
        if alias_info.backend == "vllm":
            result = await self._run_vllm_chat(alias_info, backend, messages, body)
        else:
            result = await backend.generate_chat(messages, alias_info, body)
        await self.ledger.add(alias_info.base_model, result.input_tokens, result.output_tokens)
        return self._format_chat_response(model_alias, result)

    async def handle_completion(self, body: Dict[str, Any]) -> Dict[str, Any]:
        model_alias = str(body.get("model") or "")
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("completions requires a string prompt in FlashFlow v1.")
        if body.get("stream"):
            raise ValueError("stream=true is not supported in FlashFlow v1.")
        if int(body.get("n", 1)) != 1:
            raise ValueError("FlashFlow v1 only supports n=1.")
        alias_info, backend = await self._resolve_backend(model_alias)
        if alias_info.backend == "vllm":
            result = await self._run_vllm_completion(alias_info, backend, prompt, body)
        else:
            result = await backend.generate_completion(prompt, alias_info, body)
        await self.ledger.add(alias_info.base_model, result.input_tokens, result.output_tokens)
        return self._format_completion_response(model_alias, result)

    def list_models(self) -> Dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "created": now,
                    "owned_by": "flashflow",
                }
                for alias in sorted(self.aliases.keys())
            ],
        }

    async def reset_token_usage(self) -> Dict[str, Any]:
        await self.ledger.reset()
        return {"object": "flashflow.token_usage", "data": {}}

    async def get_token_usage(self) -> Dict[str, Any]:
        return {
            "object": "flashflow.token_usage",
            "data": await self.ledger.snapshot(),
        }

    def _format_chat_response(self, model_alias: str, result: BackendResult) -> Dict[str, Any]:
        now = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": model_alias,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": int(result.input_tokens),
                "completion_tokens": int(result.output_tokens),
                "total_tokens": int(result.input_tokens + result.output_tokens),
            },
        }

    def _format_completion_response(self, model_alias: str, result: BackendResult) -> Dict[str, Any]:
        now = int(time.time())
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": now,
            "model": model_alias,
            "choices": [
                {
                    "index": 0,
                    "text": result.text,
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": int(result.input_tokens),
                "completion_tokens": int(result.output_tokens),
                "total_tokens": int(result.input_tokens + result.output_tokens),
            },
        }
