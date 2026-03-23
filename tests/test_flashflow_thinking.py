from __future__ import annotations

import torch

from flashflow.thinking import ThinkingBudgetState


def _fresh_logits(vocab_size: int = 32) -> torch.Tensor:
    return torch.zeros(vocab_size, dtype=torch.float32)


def test_thinking_budget_state_forces_cutoff_suffix_after_budget() -> None:
    state = ThinkingBudgetState(
        budget=2,
        suffix_token_ids=(7, 8, 9),
        end_think_token_ids=(11, 12),
    )

    logits = state.apply([], _fresh_logits())
    assert torch.equal(logits, _fresh_logits())

    logits = state.apply([1], _fresh_logits())
    assert torch.equal(logits, _fresh_logits())

    logits = state.apply([1, 2], _fresh_logits())
    assert logits[7].item() == 0.0
    assert torch.isneginf(logits).sum().item() == logits.numel() - 1

    logits = state.apply([1, 2, 7], _fresh_logits())
    assert logits[8].item() == 0.0
    assert torch.isneginf(logits).sum().item() == logits.numel() - 1

    logits = state.apply([1, 2, 7, 8], _fresh_logits())
    assert logits[9].item() == 0.0
    assert torch.isneginf(logits).sum().item() == logits.numel() - 1

    logits = state.apply([1, 2, 7, 8, 9], _fresh_logits())
    assert torch.equal(logits, _fresh_logits())
    assert state.seen_end_think is True
    assert state.forcing_suffix is False


def test_thinking_budget_state_stops_forcing_when_end_think_seen() -> None:
    state = ThinkingBudgetState(
        budget=10,
        suffix_token_ids=(7, 8, 9),
        end_think_token_ids=(11, 12),
    )

    logits = state.apply([3, 4, 11, 12], _fresh_logits())
    assert torch.equal(logits, _fresh_logits())
    assert state.seen_end_think is True
    assert state.forcing_suffix is False
