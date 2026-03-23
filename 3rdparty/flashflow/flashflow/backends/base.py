"""Backend interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from flashflow.types import AliasInfo, BackendResult


class BaseBackend(ABC):
    def __init__(self, model_name: str, metadata: Dict[str, Any]) -> None:
        self.model_name = str(model_name)
        self.metadata = dict(metadata)

    async def initialize(self) -> None:
        return None

    async def warmup(self) -> None:
        return None

    async def wake(self) -> None:
        return None

    async def sleep(self, level: int = 2) -> None:
        return None

    @abstractmethod
    async def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        alias_info: AliasInfo,
        request: Dict[str, Any],
    ) -> BackendResult:
        raise NotImplementedError

    @abstractmethod
    async def generate_completion(
        self,
        prompt: str,
        alias_info: AliasInfo,
        request: Dict[str, Any],
    ) -> BackendResult:
        raise NotImplementedError
