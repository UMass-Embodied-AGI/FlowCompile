from __future__ import annotations

from types import SimpleNamespace

import torch

from flowcompile.core.llm.thinking_budget import tensor_forces_token
from flowcompile.ext.vllm_plugins.thinking_budget import (
    ThinkingBudgetLogitsProcessor,
)


class _FakeTokenizer:
    def __init__(self):
        self.mapping = {
            "</think>": [7, 8],
            "CUT</think>\n\n": [4, 5, 6],
        }

    def encode(self, text, add_special_tokens=False):
        return list(self.mapping[text])


def _build_processor(monkeypatch):
    monkeypatch.setattr(
        "flowcompile.ext.vllm_plugins.thinking_budget.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(tokenizer="Qwen/Qwen3-4B", model="Qwen/Qwen3-4B")
    )
    return ThinkingBudgetLogitsProcessor(vllm_config, torch.device("cpu"), False)


def test_forces_cutoff_suffix_once_budget_is_reached(monkeypatch):
    processor = _build_processor(monkeypatch)
    params = SimpleNamespace(
        extra_args={
            "thinking_budget": 2,
            "thinking_cutoff_text": "CUT</think>\n\n",
        }
    )
    req_processor = processor.new_req_logits_processor(params)

    logits = torch.zeros(10)
    req_processor([1], logits)
    assert not tensor_forces_token(logits, 4)

    logits = torch.zeros(10)
    req_processor([1, 2], logits)
    assert tensor_forces_token(logits, 4)

    logits = torch.zeros(10)
    req_processor([1, 2, 4], logits)
    assert tensor_forces_token(logits, 5)

    logits = torch.zeros(10)
    req_processor([1, 2, 4, 5], logits)
    assert tensor_forces_token(logits, 6)

    logits = torch.zeros(10)
    req_processor([1, 2, 4, 5, 6], logits)
    assert not tensor_forces_token(logits, 6)


def test_natural_end_think_disables_budget_forcing(monkeypatch):
    processor = _build_processor(monkeypatch)
    params = SimpleNamespace(
        extra_args={
            "thinking_budget": 5,
            "thinking_cutoff_text": "CUT</think>\n\n",
        }
    )
    req_processor = processor.new_req_logits_processor(params)

    logits = torch.zeros(10)
    req_processor([1, 7, 8], logits)

    assert not tensor_forces_token(logits, 4)
