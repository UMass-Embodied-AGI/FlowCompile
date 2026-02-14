from __future__ import annotations

import pandas as pd
import pytest

from workflow_compiler.workflows.dsl_registry import get_workflow_module


def _agent_df(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "setting": [f"stub-{prefix}-a_budget_100", f"stub-{prefix}-b_budget_200"],
            "accuracy": [0.6, 0.8],
            "latency": [1.0, 2.0],
        }
    )


def test_math_backward_contract_columns():
    workflow = get_workflow_module("math")
    structure = workflow.get_full_structure()
    metrics = {
        "programmer": _agent_df("programmer"),
        "refine_solver": _agent_df("refine"),
        "detailed_solver": _agent_df("detailed"),
        "generate_solver": _agent_df("generate"),
        "sc_ensemble": _agent_df("sc"),
    }
    df = workflow.backward({"structure": structure, "metrics": metrics})
    assert not df.empty
    for col in [
        "workflow_accuracy",
        "workflow_latency",
        "structure_id",
        "programmer_setting",
        "refine_solver_setting",
        "detailed_solver_setting",
        "generate_solver_setting",
        "sc_ensemble_setting",
    ]:
        assert col in df.columns


def test_hotpotqa_backward_contract_columns():
    workflow = get_workflow_module("hotpotqa")
    structure = workflow.get_full_structure()
    metrics = {
        "answer_generate": _agent_df("answer"),
        "sc_ensemble": _agent_df("sc"),
        "format_answer": _agent_df("format"),
    }
    df = workflow.backward({"structure": structure, "metrics": metrics})
    assert not df.empty
    for col in [
        "workflow_accuracy",
        "workflow_latency",
        "structure_id",
        "answer_generate_setting",
        "sc_ensemble_setting",
        "format_answer_setting",
    ]:
        assert col in df.columns


def test_livecodebench_backward_contract_columns():
    workflow = get_workflow_module("livecodebench")
    structure = workflow.get_full_structure()
    metrics = {
        "code_generate": _agent_df("code"),
        "sc_ensemble": _agent_df("sc"),
        "reflection_test": _agent_df("reflection"),
    }
    df = workflow.backward({"structure": structure, "metrics": metrics})
    assert not df.empty
    for col in [
        "workflow_accuracy",
        "workflow_latency",
        "structure_id",
        "code_generate_setting",
        "sc_ensemble_setting",
        "reflection_test_setting",
    ]:
        assert col in df.columns


def test_livecodebench_compute_configs_requires_reflection_profile():
    workflow = get_workflow_module("livecodebench")
    with pytest.raises(ValueError, match="Missing subagent data for 'reflection_test'"):
        workflow.compute_configs(
            {
                "code_generate": _agent_df("code"),
                "sc_ensemble": _agent_df("sc"),
            },
            metadata={"show_progress": False},
        )
