from __future__ import annotations

import re
from pathlib import Path


_BANNED_IMPORT_PATTERNS = [
    re.compile(
        r"^\s*from\s+workflow_compiler\.core\.(evaluator|runners|mcp_client|memory_estimator|benchmark_registry)\b"
    ),
    re.compile(
        r"^\s*import\s+workflow_compiler\.core\.(evaluator|runners|mcp_client|memory_estimator|benchmark_registry)\b"
    ),
    re.compile(r"^\s*from\s+workflow_compiler\.core\.prompts\.optimize_prompt\b"),
    re.compile(r"^\s*import\s+workflow_compiler\.core\.prompts\.optimize_prompt\b"),
]


def test_no_imports_from_removed_core_modules():
    repo_root = Path(__file__).resolve().parents[1]
    violations = []

    for path in repo_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for pattern in _BANNED_IMPORT_PATTERNS:
                if pattern.search(line):
                    violations.append(f"{path}:{line_no}: {line.strip()}")

    assert not violations, "Banned imports found:\n" + "\n".join(violations)
