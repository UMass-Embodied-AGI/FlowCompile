from __future__ import annotations

from workflow_compiler.runtime.knn_evaluate import _build_llm_configs_for_workflow


def test_knn_hotpotqa_does_not_inject_defaults():
    llm_configs = _build_llm_configs_for_workflow(
        "hotpotqa",
        {
            "answer_generate": "qwen3-8b_budget_10",
            "sc_ensemble": "qwen3-14b_budget_2000",
        },
    )
    assert llm_configs["format_answer"] is None


def test_knn_livecodebench_test_falls_back_to_reflection_test():
    llm_configs = _build_llm_configs_for_workflow(
        "livecodebench",
        {
            "code_generate": "qwen3-8b_budget_10",
            "sc_ensemble": "qwen3-14b_budget_2000",
            "reflection_test": "gpt-4.1-mini_budget_1000",
        },
    )
    assert llm_configs["test"] == "gpt-4.1-mini_budget_1000"
