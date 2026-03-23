"""Concurrency-safe token accounting."""
from __future__ import annotations

import asyncio
import copy
from typing import Dict


class TokenLedger:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._counts: Dict[str, Dict[str, int]] = {}

    async def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        async with self._lock:
            current = self._counts.setdefault(str(model), {"input": 0, "output": 0})
            current["input"] += int(input_tokens)
            current["output"] += int(output_tokens)

    async def reset(self) -> Dict[str, Dict[str, int]]:
        async with self._lock:
            self._counts = {}
            return {}

    async def snapshot(self) -> Dict[str, Dict[str, int]]:
        async with self._lock:
            return copy.deepcopy(self._counts)
