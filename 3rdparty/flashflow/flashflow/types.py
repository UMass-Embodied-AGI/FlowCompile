"""Internal FlashFlow types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class BackendResult:
    text: str
    input_tokens: int
    output_tokens: int
    model_name: str
    finish_reason: str = "stop"


@dataclass
class AliasInfo:
    model_alias: str
    base_model: str
    budget: Any
    backend: str
    default_thinking_strategy: str
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AliasInfo":
        return cls(
            model_alias=str(data.get("model_alias") or ""),
            base_model=str(data.get("base_model") or ""),
            budget=data.get("budget"),
            backend=str(data.get("backend") or ""),
            default_thinking_strategy=str(data.get("default_thinking_strategy") or "unlimited_only"),
            metadata=dict(data),
        )


def extract_request_common(body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "temperature": body.get("temperature", 1),
        "top_p": body.get("top_p", 1),
        "max_tokens": body.get("max_tokens"),
        "stop": body.get("stop"),
    }
