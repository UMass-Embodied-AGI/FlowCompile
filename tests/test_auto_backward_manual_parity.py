from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from flowcompile.workflows.dsl_registry import get_workflow_module


def _math_metrics_multi() -> Dict[str, pd.DataFrame]:
    return {
        "programmer": pd.DataFrame(
            {
                "setting": ["prog_a", "prog_b"],
                "accuracy": [0.55, 0.75],
                "latency": [1.0, 2.0],
            }
        ),
        "refine_solver": pd.DataFrame(
            {
                "setting": ["ref_a", "ref_b"],
                "accuracy": [0.60, 0.80],
                "latency": [1.3, 2.3],
            }
        ),
        "detailed_solver": pd.DataFrame(
            {
                "setting": ["det_a", "det_b"],
                "accuracy": [0.62, 0.82],
                "latency": [1.7, 2.7],
            }
        ),
        "generate_solver": pd.DataFrame(
            {
                "setting": ["gen_a", "gen_b"],
                "accuracy": [0.50, 0.70],
                "latency": [2.1, 3.1],
            }
        ),
        "sc_ensemble": pd.DataFrame(
            {
                "setting": ["sc_a", "sc_b"],
                "accuracy": [0.90, 0.95],
                "latency": [0.4, 0.9],
            }
        ),
    }


def _hotpot_metrics_multi() -> Dict[str, pd.DataFrame]:
    return {
        "answer_generate": pd.DataFrame(
            {
                "setting": ["ans_a", "ans_b"],
                "accuracy": [0.58, 0.78],
                "latency": [1.4, 2.1],
            }
        ),
        "sc_ensemble": pd.DataFrame(
            {
                "setting": ["sc_a", "sc_b"],
                "accuracy": [0.85, 0.93],
                "latency": [0.5, 0.8],
            }
        ),
        "format_answer": pd.DataFrame(
            {
                "setting": ["fmt_a", "fmt_b"],
                "accuracy": [0.92, 0.98],
                "latency": [0.3, 0.4],
            }
        ),
    }


def _code_metrics_multi() -> Dict[str, pd.DataFrame]:
    return {
        "code_generate": pd.DataFrame(
            {
                "setting": ["code_a", "code_b"],
                "accuracy": [0.57, 0.77],
                "latency": [1.6, 2.2],
            }
        ),
        "sc_ensemble": pd.DataFrame(
            {
                "setting": ["sc_a", "sc_b"],
                "accuracy": [0.86, 0.94],
                "latency": [0.5, 0.9],
            }
        ),
        "reflection_test": pd.DataFrame(
            {
                "setting": ["fix_a", "fix_b"],
                "accuracy": [0.35, 0.55],
                "latency": [2.0, 2.8],
            }
        ),
    }


def _calculate_math_accuracy(
    p_programmer: Any = 0.0,
    p_refine: Any = 0.0,
    p_detailed: Any = 0.0,
    p_generate: Any = 0.0,
    p_sc_ensemble: Any = 1.0,
    n_programmer_refine: int = 0,
    n_detailed: int = 0,
    n_generate: int = 0,
    use_sc_ensemble: bool = False,
) -> Any:
    branch_successes = []
    if n_programmer_refine > 0:
        branch_successes.append(p_programmer * p_refine)
    if n_detailed > 0:
        branch_successes.append(p_detailed)
    if n_generate > 0:
        branch_successes.append(1 - (1 - p_generate) ** n_generate)

    if not branch_successes:
        return 0.0

    p_all_fail = 1.0
    for p_branch in branch_successes:
        p_all_fail = p_all_fail * (1 - p_branch)
    p_at_least_one = 1 - p_all_fail

    if use_sc_ensemble:
        return p_sc_ensemble * p_at_least_one
    return p_at_least_one


def _calculate_hotpot_accuracy(
    p_answer_generate: Any = 0.0,
    p_sc_ensemble: Any = 1.0,
    p_format_answer: Any = 1.0,
    n_answer_generate: int = 1,
    use_sc_ensemble: bool = False,
    use_format_answer: bool = False,
) -> Any:
    if n_answer_generate <= 0:
        return 0.0

    if use_sc_ensemble:
        p_after_generate = 1 - (1 - p_answer_generate) ** n_answer_generate
        p_after_ensemble = p_after_generate * p_sc_ensemble
    else:
        p_after_ensemble = p_answer_generate

    if use_format_answer:
        return p_after_ensemble * p_format_answer
    return p_after_ensemble


def _calculate_expected_fix_attempts(
    p_initial_correct: Any,
    p_fix_code: Any,
    max_attempts: int,
) -> Any:
    if max_attempts <= 0:
        return 0.0

    if np.any(p_initial_correct >= 1.0):
        p_initial_correct = np.clip(p_initial_correct, 0, 1)

    p_need_fix = 1 - p_initial_correct
    if isinstance(p_fix_code, (int, float)):
        if p_fix_code >= 1.0:
            expected_when_fixing = 1.0
        else:
            r = 1 - p_fix_code
            if r == 1.0:
                expected_when_fixing = max_attempts
            else:
                expected_when_fixing = (1 - r ** max_attempts) / (1 - r)
    else:
        r = 1 - p_fix_code
        epsilon = 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (1 - r ** max_attempts) / np.maximum(1 - r, epsilon)
            expected_when_fixing = np.where(
                p_fix_code >= 1.0,
                1.0,
                np.where(np.abs(1 - r) < epsilon, max_attempts, ratio),
            )
    return p_need_fix * expected_when_fixing


def _calculate_code_accuracy(
    p_code_generate: Any = 0.0,
    p_sc_ensemble: Any = 1.0,
    p_fix_code: Any = 0.0,
    n_code_generate: int = 1,
    use_sc_ensemble: bool = False,
    max_test_attempts: int = 0,
) -> Any:
    if n_code_generate <= 0:
        return 0.0

    if use_sc_ensemble:
        p_at_least_one_gen = 1 - (1 - p_code_generate) ** n_code_generate
        p_initial_correct = p_at_least_one_gen * p_sc_ensemble
    else:
        p_initial_correct = p_code_generate

    if max_test_attempts <= 0:
        return p_initial_correct

    p_fix_success = 1 - (1 - p_fix_code) ** max_test_attempts
    return p_initial_correct + (1 - p_initial_correct) * p_fix_success


def _manual_math_backward(workflow, payload: Dict[str, Any]) -> pd.DataFrame:
    ctx = workflow.metric_context(payload)

    n_prog = ctx.count("programmer")
    n_refine = ctx.count("refine_solver")
    n_prog_refine = min(n_prog, n_refine)
    n_detailed = ctx.count("detailed_solver")
    n_generate = ctx.count("generate_solver")
    use_sc = ctx.enabled("sc_ensemble")

    workflow_latency = (
        ctx.lat("programmer", 0.0) * n_prog
        + ctx.lat("refine_solver", 0.0) * n_refine
        + ctx.lat("detailed_solver", 0.0) * n_detailed
        + ctx.lat("generate_solver", 0.0) * n_generate
        + ctx.lat("sc_ensemble", 0.0) * ctx.count("sc_ensemble")
    )
    workflow_accuracy = _calculate_math_accuracy(
        p_programmer=ctx.acc("programmer", 0.0),
        p_refine=ctx.acc("refine_solver", 0.0),
        p_detailed=ctx.acc("detailed_solver", 0.0),
        p_generate=ctx.acc("generate_solver", 0.0),
        p_sc_ensemble=ctx.acc("sc_ensemble", 1.0),
        n_programmer_refine=n_prog_refine,
        n_detailed=n_detailed,
        n_generate=n_generate,
        use_sc_ensemble=use_sc,
    )
    return ctx.finish(
        workflow_accuracy=workflow_accuracy,
        workflow_latency=workflow_latency,
    )


def _manual_hotpot_backward(workflow, payload: Dict[str, Any]) -> pd.DataFrame:
    ctx = workflow.metric_context(payload)

    n_generate = ctx.count("answer_generate")
    use_sc = ctx.enabled("sc_ensemble")
    use_format = ctx.enabled("format_answer")

    workflow_accuracy = _calculate_hotpot_accuracy(
        p_answer_generate=ctx.acc("answer_generate", 0.0),
        p_sc_ensemble=ctx.acc("sc_ensemble", 1.0),
        p_format_answer=ctx.acc("format_answer", 1.0),
        n_answer_generate=n_generate,
        use_sc_ensemble=use_sc,
        use_format_answer=use_format,
    )
    workflow_latency = (
        ctx.lat("answer_generate", 0.0) * n_generate
        + ctx.lat("sc_ensemble", 0.0) * ctx.count("sc_ensemble")
        + ctx.lat("format_answer", 0.0) * ctx.count("format_answer")
    )
    return ctx.finish(
        workflow_accuracy=workflow_accuracy,
        workflow_latency=workflow_latency,
    )


def _manual_code_backward(workflow, payload: Dict[str, Any]) -> pd.DataFrame:
    ctx = workflow.metric_context(payload)

    n_code = ctx.count("code_generate")
    use_sc = ctx.enabled("sc_ensemble")
    n_fix_attempts = ctx.count("reflection_test")

    if use_sc:
        p_at_least_one_gen = 1 - (1 - ctx.acc("code_generate", 0.0)) ** n_code
        p_initial_correct = p_at_least_one_gen * ctx.acc("sc_ensemble", 1.0)
    else:
        p_initial_correct = ctx.acc("code_generate", 0.0)

    workflow_accuracy = _calculate_code_accuracy(
        p_code_generate=ctx.acc("code_generate", 0.0),
        p_sc_ensemble=ctx.acc("sc_ensemble", 1.0),
        p_fix_code=ctx.acc("reflection_test", 0.0),
        n_code_generate=n_code,
        use_sc_ensemble=use_sc,
        max_test_attempts=n_fix_attempts,
    )

    workflow_latency = (
        ctx.lat("code_generate", 0.0) * n_code
        + ctx.lat("sc_ensemble", 0.0) * ctx.count("sc_ensemble")
    )
    if n_fix_attempts > 0:
        expected_fix_attempts = _calculate_expected_fix_attempts(
            p_initial_correct=p_initial_correct,
            p_fix_code=ctx.acc("reflection_test", 0.0),
            max_attempts=n_fix_attempts,
        )
        workflow_latency = workflow_latency + expected_fix_attempts * ctx.lat("reflection_test", 0.0)

    return ctx.finish(
        workflow_accuracy=workflow_accuracy,
        workflow_latency=workflow_latency,
    )


def _assert_equal_metrics(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    setting_cols = sorted([col for col in actual.columns if col.endswith("_setting")])
    sort_cols = setting_cols + ["structure_id"]

    actual_sorted = actual.sort_values(sort_cols).reset_index(drop=True)
    expected_sorted = expected.sort_values(sort_cols).reset_index(drop=True)

    assert len(actual_sorted) == len(expected_sorted)
    np.testing.assert_allclose(
        actual_sorted["workflow_accuracy"].to_numpy(),
        expected_sorted["workflow_accuracy"].to_numpy(),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_sorted["workflow_latency"].to_numpy(),
        expected_sorted["workflow_latency"].to_numpy(),
        atol=1e-10,
        rtol=1e-10,
    )


def test_auto_backward_matches_original_manual_math():
    workflow = get_workflow_module("math")
    metrics = _math_metrics_multi()
    for structure in workflow.enumerate_structures():
        payload = {"structure": structure, "metrics": metrics}
        expected = _manual_math_backward(workflow, payload)
        actual = workflow.backward(payload)
        _assert_equal_metrics(actual, expected)


def test_auto_backward_matches_original_manual_hotpotqa():
    workflow = get_workflow_module("hotpotqa")
    metrics = _hotpot_metrics_multi()
    for structure in workflow.enumerate_structures():
        payload = {"structure": structure, "metrics": metrics}
        expected = _manual_hotpot_backward(workflow, payload)
        actual = workflow.backward(payload)
        _assert_equal_metrics(actual, expected)


def test_auto_backward_matches_original_manual_livecodebench():
    workflow = get_workflow_module("livecodebench")
    metrics = _code_metrics_multi()
    for structure in workflow.enumerate_structures():
        payload = {"structure": structure, "metrics": metrics}
        expected = _manual_code_backward(workflow, payload)
        actual = workflow.backward(payload)
        _assert_equal_metrics(actual, expected)
