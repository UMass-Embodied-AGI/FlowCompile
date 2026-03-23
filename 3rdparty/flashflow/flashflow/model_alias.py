"""Model alias helpers for FlashFlow."""
from __future__ import annotations

from typing import Any, Optional, Tuple


def parse_model_alias(alias: str) -> Tuple[Optional[str], Any]:
    if alias is None:
        return None, None
    text = str(alias)
    if "_budget_" not in text:
        return text, None
    model_name, budget_text = text.split("_budget_", 1)
    if budget_text in {"unlimited", "nothinking"}:
        return model_name, budget_text
    try:
        return model_name, int(budget_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid budget alias '{alias}'. Expected int, 'unlimited', or 'nothinking'."
        ) from exc


def build_model_alias(model: str, budget: Any) -> str:
    if budget is None:
        return str(model)
    return f"{model}_budget_{budget}"
