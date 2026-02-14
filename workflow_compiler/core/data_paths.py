"""Helpers for resolving dataset file paths."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def candidate_data_paths(path: Optional[str]) -> List[str]:
    """Return normalized candidate paths, enforcing `data/...` layout."""
    if not path:
        return []

    raw = str(path)
    normalized = raw.replace("\\", "/")

    candidates: List[str] = []
    seen = set()

    def add(value: str) -> None:
        if value and value not in seen:
            candidates.append(value)
            seen.add(value)

    add(raw)

    if "data/ours/" in normalized:
        add(raw.replace("data/ours/", "data/", 1))

    return candidates


def resolve_existing_path(path: Optional[str]) -> Optional[str]:
    """Return the first existing candidate path, or None if none exist."""
    for candidate in candidate_data_paths(path):
        if Path(candidate).exists():
            return candidate
    return None


def resolve_required_path(path: Optional[str], *, label: str) -> str:
    """
    Resolve a required path across known layout variants.

    Raises FileNotFoundError with candidate details when unresolved.
    """
    resolved = resolve_existing_path(path)
    if resolved:
        return resolved

    candidates = candidate_data_paths(path)
    raise FileNotFoundError(
        f"{label} not found. Checked: {candidates if candidates else [path]}"
    )
