from __future__ import annotations

import asyncio

import pytest

from flowcompile.dsl.executor import DslExecutor
from flowcompile.dsl import runtime


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


def test_executor_aclose_closes_agent_llms():
    class _ClosableLLM:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    class _Agent:
        def __init__(self, llm):
            self.llm = llm

    llm_a = _ClosableLLM()
    llm_b = _ClosableLLM()
    executor = DslExecutor(
        spec={"nodes": [], "edges": [], "entry": None},
        workflow_type="math",
        config={},
    )
    executor._agent_instances = {
        "a": _Agent(llm_a),
        "b": _Agent(llm_b),
    }

    asyncio.run(executor.aclose())

    assert llm_a.closed is True
    assert llm_b.closed is True
    assert executor._agent_instances == {}


def test_run_dsl_query_closes_executor_even_on_error(monkeypatch, tmp_path):
    state = {"closed": False}

    class _FakeWorkflowModule:
        def compile(self):
            return {"nodes": [], "edges": [], "entry": None}

        def get_full_structure(self):
            return {"structure_id": "stub", "total_branches": 1, "is_full": True, "active_agent_counts": {}}

    class _FakeExecutor:
        def __init__(self, spec, workflow_type, config):
            self.spec = spec
            self.workflow_type = workflow_type
            self.config = config

        async def run(self, inputs):
            raise RuntimeError("executor boom")

        async def aclose(self):
            state["closed"] = True

    monkeypatch.setattr(runtime, "get_workflow_module", lambda *_args, **_kwargs: _FakeWorkflowModule())
    monkeypatch.setattr(runtime, "apply_structure", lambda spec, *_args, **_kwargs: spec)
    monkeypatch.setattr(runtime, "DslExecutor", _FakeExecutor)

    with pytest.raises(RuntimeError, match="executor boom"):
        asyncio.run(
            runtime.run_dsl_query(
                query={"problem": "1+1?", "question_id": "q1"},
                config={},
                workflow_type="math",
                output_dir=tmp_path,
            )
        )

    assert state["closed"] is True
