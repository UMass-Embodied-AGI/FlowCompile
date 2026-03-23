"""Thinking-budget helpers for FlashFlow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


DEFAULT_THINKING_BUDGET_CUTOFF_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)
THINK_END_TEXT = "</think>"
THINKING_BUDGET_EXTRA_ARG = "flashflow_thinking_budget"
THINKING_SUFFIX_TOKEN_IDS_ARG = "flashflow_suffix_token_ids"
THINKING_END_TOKEN_IDS_ARG = "flashflow_end_think_token_ids"


def _endswith_token_ids(output_ids: Sequence[int], suffix_ids: Sequence[int]) -> bool:
    if not suffix_ids or len(output_ids) < len(suffix_ids):
        return False
    return list(output_ids[-len(suffix_ids):]) == list(suffix_ids)


def force_token_id(logits: Any, token_id: int):
    logits.fill_(float("-inf"))
    logits[token_id] = 0.0
    return logits


@dataclass
class ThinkingBudgetState:
    budget: int
    suffix_token_ids: tuple[int, ...]
    end_think_token_ids: tuple[int, ...]
    reasoning_tokens_emitted: int = 0
    seen_end_think: bool = False
    forcing_suffix: bool = False
    suffix_position: int = 0
    processed_tokens: int = 0

    def sync_output_ids(self, output_ids: Sequence[int]) -> None:
        start = min(self.processed_tokens, len(output_ids))
        for pos in range(start, len(output_ids)):
            token_id = int(output_ids[pos])
            self._consume_token(output_ids[: pos + 1], token_id)
        self.processed_tokens = len(output_ids)
        if (
            not self.seen_end_think
            and not self.forcing_suffix
            and self.reasoning_tokens_emitted >= self.budget
        ):
            self.forcing_suffix = True
            self.suffix_position = 0

    def _consume_token(self, output_prefix: Sequence[int], token_id: int) -> None:
        if self.forcing_suffix:
            expected_token = self.suffix_token_ids[self.suffix_position]
            if token_id != expected_token:
                raise ValueError("Forced thinking cutoff suffix diverged from generated output.")
            self.suffix_position += 1
            if self.suffix_position >= len(self.suffix_token_ids):
                self.forcing_suffix = False
                self.seen_end_think = True
            return
        if self.seen_end_think:
            return
        self.reasoning_tokens_emitted += 1
        if _endswith_token_ids(output_prefix, self.end_think_token_ids):
            self.seen_end_think = True
            self.reasoning_tokens_emitted = max(
                0,
                self.reasoning_tokens_emitted - len(self.end_think_token_ids),
            )

    def apply(self, output_ids: Sequence[int], logits: Any):
        self.sync_output_ids(output_ids)
        if self.forcing_suffix and self.suffix_position < len(self.suffix_token_ids):
            return force_token_id(logits, self.suffix_token_ids[self.suffix_position])
        return logits


def _coerce_token_id_sequence(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of token ids.")
    try:
        return tuple(int(item) for item in value)
    except Exception as exc:
        raise ValueError(f"{name} must only contain integer token ids.") from exc


class FlashFlowRequestThinkingBudgetProcessor:
    """Request-scoped thinking-budget processor used through vLLM's adapter API."""

    def __init__(
        self,
        budget: int,
        suffix_token_ids: Sequence[int],
        end_think_token_ids: Sequence[int],
    ) -> None:
        self._state = ThinkingBudgetState(
            budget=int(budget),
            suffix_token_ids=tuple(int(token_id) for token_id in suffix_token_ids),
            end_think_token_ids=tuple(int(token_id) for token_id in end_think_token_ids),
        )

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        return self._state.apply(output_ids, logits)


class FlashFlowThinkingBudgetLogitsProcessor(AdapterLogitsProcessor):
    """Engine-level vLLM logits processor for FlashFlow thinking budgets."""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra_args = params.extra_args or {}
        budget = extra_args.get(THINKING_BUDGET_EXTRA_ARG)
        if budget is None:
            return
        if not isinstance(budget, int) or budget < 0:
            raise ValueError(
                f"{THINKING_BUDGET_EXTRA_ARG} must be a non-negative integer; got {budget!r}."
            )
        _coerce_token_id_sequence(
            THINKING_SUFFIX_TOKEN_IDS_ARG,
            extra_args.get(THINKING_SUFFIX_TOKEN_IDS_ARG),
        )
        _coerce_token_id_sequence(
            THINKING_END_TOKEN_IDS_ARG,
            extra_args.get(THINKING_END_TOKEN_IDS_ARG),
        )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        self.validate_params(params)
        extra_args = params.extra_args or {}
        budget = extra_args.get(THINKING_BUDGET_EXTRA_ARG)
        if budget is None:
            return None
        return FlashFlowRequestThinkingBudgetProcessor(
            budget=int(budget),
            suffix_token_ids=_coerce_token_id_sequence(
                THINKING_SUFFIX_TOKEN_IDS_ARG,
                extra_args.get(THINKING_SUFFIX_TOKEN_IDS_ARG),
            ),
            end_think_token_ids=_coerce_token_id_sequence(
                THINKING_END_TOKEN_IDS_ARG,
                extra_args.get(THINKING_END_TOKEN_IDS_ARG),
            ),
        )
