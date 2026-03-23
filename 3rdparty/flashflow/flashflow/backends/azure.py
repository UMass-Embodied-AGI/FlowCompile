"""Azure backend support for FlashFlow."""
from __future__ import annotations

from typing import Any, Dict, List

from flashflow.backends.base import BaseBackend
from flashflow.types import AliasInfo, BackendResult, extract_request_common

try:
    from openai import AsyncAzureOpenAI
except Exception:  # pragma: no cover - optional dependency resolution
    AsyncAzureOpenAI = None


class AzureOpenAIBackend(BaseBackend):
    def __init__(self, model_name: str, metadata: Dict[str, Any]) -> None:
        super().__init__(model_name, metadata)
        self._client = None
        self._request_model = (
            metadata.get("azure_deployment")
            or metadata.get("deployment_name")
            or metadata.get("deployment")
            or model_name
        )

    async def initialize(self) -> None:
        if AsyncAzureOpenAI is None:
            raise ImportError("AsyncAzureOpenAI is unavailable. Please install the openai package.")
        self._client = AsyncAzureOpenAI(
            azure_endpoint=self.metadata.get("azure_endpoint"),
            api_key=self.metadata.get("api_key"),
            api_version=self.metadata.get("api_version") or "2024-02-15-preview",
        )

    def _validate_alias(self, alias_info: AliasInfo) -> None:
        if alias_info.budget not in (None, "unlimited"):
            raise ValueError(
                f"Azure-backed model '{alias_info.base_model}' only supports unlimited thinking mode in FlashFlow."
            )

    async def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        alias_info: AliasInfo,
        request: Dict[str, Any],
    ) -> BackendResult:
        self._validate_alias(alias_info)
        options = extract_request_common(request)
        max_tokens = options.pop("max_tokens", None)
        if max_tokens is not None:
            options["max_completion_tokens"] = max_tokens
        response = await self._client.chat.completions.create(
            model=self._request_model,
            messages=messages,
            **{k: v for k, v in options.items() if v is not None},
        )
        usage = getattr(response, "usage", None)
        content = response.choices[0].message.content or ""
        return BackendResult(
            text=content.split("</think>")[-1].strip() if content else "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model_name=self.model_name,
            finish_reason=str(response.choices[0].finish_reason or "stop"),
        )

    async def generate_completion(
        self,
        prompt: str,
        alias_info: AliasInfo,
        request: Dict[str, Any],
    ) -> BackendResult:
        return await self.generate_chat(
            [{"role": "user", "content": prompt}],
            alias_info,
            request,
        )
