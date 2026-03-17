from __future__ import annotations

import pandas as pd
import pytest

from workflow_compiler.core.analysis.prediction import (
    SearchSpaceSpec,
    apply_search_space_to_subagents,
    apply_structure_constraints,
)


def _build_math_subagents() -> dict[str, pd.DataFrame]:
    settings = [
        ("qwen3-4b_budget_100", 0.8, 1.0),
        ("qwen3-8b_budget_200", 0.9, 2.0),
    ]
    df = pd.DataFrame(settings, columns=["setting", "accuracy", "latency"])
    return {
        "programmer": df.copy(),
        "refine_solver": df.copy(),
        "detailed_solver": df.copy(),
        "generate_solver": df.copy(),
        "sc_ensemble": df.copy(),
    }


def test_search_space_filters_models_and_budgets():
    spec = SearchSpaceSpec.from_dict(
        {
            "search_axes": ["model", "budget", "structure"],
            "models": ["qwen3-4b"],
            "budgets": ["100"],
        }
    )
    filtered, info = apply_search_space_to_subagents(
        _build_math_subagents(),
        required_agents=["programmer", "refine_solver", "detailed_solver", "generate_solver", "sc_ensemble"],
        spec=spec,
    )
    for agent, df in filtered.items():
        assert len(df) == 1, f"expected single row for {agent}"
        assert df["setting"].iloc[0] == "qwen3-4b_budget_100"
    assert "resolved_locks" in info


def test_search_space_requires_budget_lock_when_budget_axis_disabled():
    spec = SearchSpaceSpec.from_dict(
        {
            "search_axes": ["model", "structure"],
        }
    )
    with pytest.raises(ValueError, match="Budget axis disabled"):
        apply_search_space_to_subagents(
            _build_math_subagents(),
            required_agents=["programmer", "refine_solver", "detailed_solver", "generate_solver", "sc_ensemble"],
            spec=spec,
        )


def test_structure_lock_validation():
    structures = [
        {"structure_id": "s__programmer-c1__refine_solver-c1__detailed_solver-c1__generate_solver-c2__sc_ensemble-c1"},
        {"structure_id": "s__programmer-c1__refine_solver-c0__detailed_solver-c1__generate_solver-c2__sc_ensemble-c1"},
    ]
    spec = SearchSpaceSpec.from_dict(
        {
            "search_axes": ["model", "budget"],
            "structures": ["s__programmer-c1__refine_solver-c1__detailed_solver-c1__generate_solver-c2__sc_ensemble-c1"],
        }
    )
    filtered, info = apply_structure_constraints(structures, spec)
    assert len(filtered) == 1
    assert filtered[0]["structure_id"] == "s__programmer-c1__refine_solver-c1__detailed_solver-c1__generate_solver-c2__sc_ensemble-c1"
    assert info["resolved_structure"] == "s__programmer-c1__refine_solver-c1__detailed_solver-c1__generate_solver-c2__sc_ensemble-c1"


def test_single_branch_multi_node_kept_by_default():
    """A single-branch structure with 2+ nodes should survive default filtering."""
    structures = [
        {
            "structure_id": "full",
            "active_node_ids": ["select_efficiency", "summarize_pdfs"],
            "total_branches": 1,
        },
    ]
    spec = SearchSpaceSpec()
    filtered, info = apply_structure_constraints(structures, spec)
    assert len(filtered) == 1
    assert filtered[0]["structure_id"] == "full"
    assert info.get("exclude_single_node_default") is True


def test_single_node_dropped_when_multi_node_exists():
    """A single-node structure should be dropped when multi-node structures exist."""
    structures = [
        {
            "structure_id": "full",
            "active_node_ids": ["a", "b"],
            "total_branches": 2,
        },
        {
            "structure_id": "pruned",
            "active_node_ids": ["a"],
            "total_branches": 1,
        },
    ]
    spec = SearchSpaceSpec()
    filtered, info = apply_structure_constraints(structures, spec)
    assert len(filtered) == 1
    assert filtered[0]["structure_id"] == "full"


def test_single_node_only_workflow_not_filtered():
    """When ALL structures have <=1 node, none should be dropped."""
    structures = [
        {
            "structure_id": "only_one",
            "active_node_ids": ["a"],
            "total_branches": 1,
        },
    ]
    spec = SearchSpaceSpec()
    filtered, info = apply_structure_constraints(structures, spec)
    assert len(filtered) == 1
    assert filtered[0]["structure_id"] == "only_one"

