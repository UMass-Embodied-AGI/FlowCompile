from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

DEFAULT_THINKING_BUDGET_CUTOFF_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)
DEFAULT_THINKING_BUDGET_REASONING_PARSER = "qwen3"
DEFAULT_THINKING_BUDGET_VLLM_ARG_NAME = "thinking_budget"
THINKING_CUTOFF_TEXT_ARG_NAME = "thinking_cutoff_text"
THINKING_BUDGET_ARG_NAME_ARG_NAME = "thinking_budget_arg_name"
THINKING_BUDGET_HF_MODEL_ARG_NAME = "thinking_budget_hf_model_name"
THINK_END_TEXT = "</think>"


def force_token_id(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    logits.fill_(float("-inf"))
    logits[token_id] = 0.0
    return logits


def tensor_forces_token(logits: torch.Tensor, token_id: int) -> bool:
    forced_value = logits[token_id].item()
    return all(
        (idx == token_id and value.item() == forced_value)
        or (idx != token_id and value.item() == float("-inf"))
        for idx, value in enumerate(logits)
    )


def _endswith_token_ids(output_ids: Sequence[int], suffix_ids: Sequence[int]) -> bool:
    if not suffix_ids or len(output_ids) < len(suffix_ids):
        return False
    return list(output_ids[-len(suffix_ids):]) == list(suffix_ids)


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
                raise ValueError(
                    "Forced thinking cutoff suffix diverged from generated output."
                )
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
                0, self.reasoning_tokens_emitted - len(self.end_think_token_ids)
            )

    def apply(self, output_ids: Sequence[int], logits: torch.Tensor) -> torch.Tensor:
        self.sync_output_ids(output_ids)
        if self.forcing_suffix and self.suffix_position < len(self.suffix_token_ids):
            return force_token_id(logits, self.suffix_token_ids[self.suffix_position])
        return logits
