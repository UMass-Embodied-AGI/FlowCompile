from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from flowcompile.core.analysis.prediction import calculate_workflow_accuracy, calculate_workflow_metrics
from flowcompile.dsl.runtime import DslWorkflowRunner
from flowcompile.workflows.dsl_registry import get_workflow_module


def test_workflow_forward_signature_is_unified():
    expected = ("query",)

    for workflow_type in ("math", "hotpotqa", "livecodebench"):
        workflow = get_workflow_module(workflow_type)
        params = tuple(inspect.signature(workflow.forward).parameters.keys())
        assert params == expected


def test_workflow_backward_signature_is_unified():
    expected = ("payload",)

    for workflow_type in ("math", "hotpotqa", "livecodebench"):
        workflow = get_workflow_module(workflow_type)
        params = tuple(inspect.signature(workflow.backward).parameters.keys())
        assert params == expected


def test_workflow_outputs_are_unified():
    expected_keys = {"final_answer", "full_solution", "final_solution"}

    for workflow_type in ("math", "hotpotqa", "livecodebench"):
        spec = get_workflow_module(workflow_type).compile()
        assert expected_keys.issubset(set(spec.get("outputs", {}).keys()))


def _collect_input_refs(obj):
    refs = []
    if isinstance(obj, dict):
        ref = obj.get("ref")
        if isinstance(ref, str) and ref.startswith("input."):
            refs.append(ref)
        for value in obj.values():
            refs.extend(_collect_input_refs(value))
    elif isinstance(obj, list):
        for value in obj:
            refs.extend(_collect_input_refs(value))
    return refs


def test_workflow_compile_uses_query_input_namespace():
    for workflow_type in ("math", "hotpotqa", "livecodebench"):
        spec = get_workflow_module(workflow_type).compile()
        refs = _collect_input_refs(spec)
        assert refs
        assert all(ref.startswith("input.query.") for ref in refs)


def test_dsl_runner_rejects_non_dict_query(tmp_path: Path):
    runner = DslWorkflowRunner(
        name="test_runner",
        llm_configs={},
        workflow_type="math",
        output_dir=tmp_path,
    )
    with pytest.raises(TypeError, match="dict payload"):
        asyncio.run(runner("x + y"))


def test_dsl_runner_rejects_legacy_positional_args(tmp_path: Path):
    runner = DslWorkflowRunner(
        name="test_runner",
        llm_configs={},
        workflow_type="math",
        output_dir=tmp_path,
    )
    with pytest.raises(TypeError):
        runner({"problem": "x + y"}, "entry_point", "question_id")


def test_workflow_backward_rejects_non_dict_payload():
    workflow = get_workflow_module("math")
    with pytest.raises(TypeError, match="dict"):
        workflow.backward("not-a-dict")


def test_workflow_backward_requires_structure():
    workflow = get_workflow_module("math")
    with pytest.raises(ValueError, match="structure"):
        workflow.backward({})


def test_workflow_metrics_helper_accepts_one_payload():
    workflow = get_workflow_module("math")
    structure = workflow.get_full_structure()
    payload = {
        "workflow_type": "math",
        "structure": structure,
        "metrics": {
            "programmer": _tiny_agent_df("programmer"),
            "refine_solver": _tiny_agent_df("refine"),
            "detailed_solver": _tiny_agent_df("detailed"),
            "generate_solver": _tiny_agent_df("generate"),
            "sc_ensemble": _tiny_agent_df("sc"),
        },
    }
    result = calculate_workflow_metrics(payload)
    assert not result.empty
    assert "workflow_accuracy" in result.columns


def test_workflow_accuracy_helper_accepts_one_payload():
    workflow = get_workflow_module("hotpotqa")
    structure = workflow.get_full_structure()
    payload = {
        "workflow_type": "hotpotqa",
        "structure": structure,
        "p_answer_generate": 0.7,
        "p_sc_ensemble": 0.8,
        "p_format_answer": 0.9,
    }
    value = calculate_workflow_accuracy(payload)
    assert isinstance(value, float)


def _tiny_agent_df(prefix: str):
    import pandas as pd

    return pd.DataFrame(
        {
            "setting": [f"stub-{prefix}-budget-100"],
            "accuracy": [0.7],
            "latency": [1.0],
        }
    )
