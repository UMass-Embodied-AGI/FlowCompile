from __future__ import annotations

import pytest

from flowcompile.compiler.validation import (
    _active_llm_refs_for_structure,
    _build_evaluation_items,
    _build_llm_configs_for_workflow,
    _sample_pareto_even_by_latency,
    _validate_active_llm_refs,
)
from flowcompile.workflows.dsl_registry import get_workflow_module


def _cfg(latency: float, rank: int) -> dict:
    return {
        "metrics": {"expected_latency": latency},
        "pareto": {"rank": rank, "is_pareto": True},
    }


def test_sample_pareto_even_by_latency_includes_endpoints():
    configs = [_cfg(1.0, 1), _cfg(2.0, 2), _cfg(100.0, 3), _cfg(101.0, 4), _cfg(102.0, 5)]
    sampled, meta = _sample_pareto_even_by_latency(configs, 3)
    sampled_latencies = [c["metrics"]["expected_latency"] for c in sampled]

    assert sampled_latencies[0] == 1.0
    assert sampled_latencies[-1] == 102.0
    assert 100.0 in sampled_latencies
    assert meta["selected_count"] == 3
    assert meta["candidate_count"] == 5


def test_sample_pareto_even_by_latency_n1_picks_min_latency():
    configs = [_cfg(5.0, 3), _cfg(3.0, 2), _cfg(7.0, 4)]
    sampled, meta = _sample_pareto_even_by_latency(configs, 1)

    assert len(sampled) == 1
    assert sampled[0]["metrics"]["expected_latency"] == 3.0
    assert meta["selected_count"] == 1


def test_sample_pareto_even_by_latency_n_ge_total_keeps_all():
    configs = [_cfg(1.0, 1), _cfg(2.0, 2), _cfg(3.0, 3)]
    sampled, meta = _sample_pareto_even_by_latency(configs, 10)

    assert len(sampled) == 3
    assert [c["metrics"]["expected_latency"] for c in sampled] == [1.0, 2.0, 3.0]
    assert meta["selected_count"] == 3


def test_sample_pareto_even_by_latency_requires_predicted_latency():
    with pytest.raises(ValueError, match="Missing predicted latency"):
        _sample_pareto_even_by_latency([{"is_pareto": True}], 1)


def test_build_llm_configs_hotpotqa_does_not_inject_defaults():
    llm_configs = _build_llm_configs_for_workflow(
        "hotpotqa",
        {
            "agents": {
                "answer_generate": {"setting": "qwen3-8b_budget_10"},
                "sc_ensemble": {"setting": "qwen3-14b_budget_2000"},
            }
        },
    )
    assert "format_answer" not in llm_configs


def test_validate_active_llm_refs_allows_pruned_operators():
    workflow = get_workflow_module("hotpotqa")
    structure_id = next(
        structure["structure_id"]
        for structure in workflow.enumerate_structures()
        if int((structure.get("active_agent_counts") or {}).get("sc_ensemble", 0)) > 0
        and int((structure.get("active_agent_counts") or {}).get("format_answer", 0)) == 0
    )
    llm_configs = _build_llm_configs_for_workflow(
        "hotpotqa",
        {
            "agents": {
                "answer_generate": {"setting": "qwen3-8b_budget_10"},
                "sc_ensemble": {"setting": "qwen3-14b_budget_2000"},
            }
        },
    )
    active_refs = _active_llm_refs_for_structure("hotpotqa", structure_id)

    assert "format_answer" not in active_refs
    _validate_active_llm_refs(
        workflow_type="hotpotqa",
        structure_id=structure_id,
        llm_configs=llm_configs,
        config={"structure_id": structure_id},
        config_idx=0,
        active_llm_refs=active_refs,
    )


def test_validate_active_llm_refs_rejects_missing_active_operator():
    workflow = get_workflow_module("hotpotqa")
    structure_id = next(
        structure["structure_id"]
        for structure in workflow.enumerate_structures()
        if int((structure.get("active_agent_counts") or {}).get("format_answer", 0)) > 0
        and int((structure.get("active_agent_counts") or {}).get("sc_ensemble", 0)) == 0
    )
    llm_configs = _build_llm_configs_for_workflow(
        "hotpotqa",
        {
            "agents": {
                "answer_generate": {"setting": "qwen3-8b_budget_10"},
            }
        },
    )
    active_refs = _active_llm_refs_for_structure("hotpotqa", structure_id)

    with pytest.raises(ValueError, match="format_answer"):
        _validate_active_llm_refs(
            workflow_type="hotpotqa",
            structure_id=structure_id,
            llm_configs=llm_configs,
            config={"structure_id": structure_id},
            config_idx=0,
            active_llm_refs=active_refs,
        )


def test_build_llm_configs_livecodebench_uses_agents_map():
    llm_configs = _build_llm_configs_for_workflow(
        "livecodebench",
        {
            "agents": {
                "code_generate": {"setting": "qwen3-8b_budget_10"},
                "sc_ensemble": {"setting": "qwen3-14b_budget_2000"},
                "reflection_test": {"setting": "gpt-4.1-mini_budget_1000"},
            }
        },
    )
    assert llm_configs["reflection_test"] == "gpt-4.1-mini_budget_1000"
    assert "test" not in llm_configs


def test_build_evaluation_items_uses_pareto_rank_as_config_index():
    items = _build_evaluation_items(
        [
            {"pareto": {"rank": 5, "is_pareto": True}},
            {"pareto": {"rank": 0, "is_pareto": True}},
            {"pareto": {"rank": 2, "is_pareto": True}},
        ]
    )
    assert [idx for idx, _ in items] == [5, 0, 2]


def test_build_evaluation_items_falls_back_when_rank_missing_or_duplicated():
    items = _build_evaluation_items(
        [
            {"pareto": {"rank": 0, "is_pareto": True}},
            {"pareto": {"is_pareto": True}},
            {"pareto": {"rank": 0, "is_pareto": True}},
        ]
    )
    indices = [idx for idx, _ in items]
    assert len(set(indices)) == 3
    assert indices[0] == 0
