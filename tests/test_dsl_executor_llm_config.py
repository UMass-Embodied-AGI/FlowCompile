from __future__ import annotations

import pytest

from workflow_compiler.dsl.executor import DslExecutor


def test_executor_raises_when_active_operator_setting_missing():
    executor = DslExecutor(
        spec={"nodes": [], "edges": [], "entry": None},
        workflow_type="hotpotqa",
        config={
            "agents": {
                "answer_generate": {"setting": "qwen3-8b_budget_10"},
            }
        },
    )

    with pytest.raises(ValueError, match="format_answer"):
        executor._build_llm("format_answer")
