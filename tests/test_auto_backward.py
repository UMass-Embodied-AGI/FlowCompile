from __future__ import annotations

import importlib
from typing import Dict

import pandas as pd
import pytest

from flowcompile.dsl.torchlike import AgentNode, WorkflowModule
from flowcompile.workflows.dsl_registry import get_workflow_module


def _single_df(setting: str, accuracy: float, latency: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "setting": [setting],
            "accuracy": [accuracy],
            "latency": [latency],
        }
    )


def _math_expected_accuracy(
    p_programmer: float,
    p_refine: float,
    p_detailed: float,
    p_generate: float,
    p_sc_ensemble: float,
    n_programmer_refine: int,
    n_detailed: int,
    n_generate: int,
    use_sc_ensemble: bool,
) -> float:
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


def _hotpot_expected_accuracy(
    p_answer_generate: float,
    p_sc_ensemble: float,
    p_format_answer: float,
    n_answer_generate: int,
    use_sc_ensemble: bool,
    use_format_answer: bool,
) -> float:
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


def _expected_fix_attempts(p_initial_correct: float, p_fix_code: float, max_attempts: int) -> float:
    if max_attempts <= 0:
        return 0.0
    if p_fix_code >= 1.0:
        expected_when_fixing = 1.0
    else:
        r = 1 - p_fix_code
        if r == 1.0:
            expected_when_fixing = float(max_attempts)
        else:
            expected_when_fixing = (1 - r ** max_attempts) / (1 - r)
    return (1 - p_initial_correct) * expected_when_fixing


def _code_expected_accuracy(
    p_code_generate: float,
    p_sc_ensemble: float,
    p_fix_code: float,
    n_code_generate: int,
    use_sc_ensemble: bool,
    max_test_attempts: int,
) -> float:
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


def _math_metrics() -> Dict[str, pd.DataFrame]:
    return {
        "programmer": _single_df("prog", 0.61, 1.1),
        "refine_solver": _single_df("refine", 0.72, 2.2),
        "detailed_solver": _single_df("detailed", 0.83, 3.3),
        "generate_solver": _single_df("generate", 0.54, 4.4),
        "sc_ensemble": _single_df("sc", 0.91, 5.5),
    }


def _hotpot_metrics() -> Dict[str, pd.DataFrame]:
    return {
        "answer_generate": _single_df("answer", 0.63, 1.7),
        "sc_ensemble": _single_df("sc", 0.88, 2.8),
        "format_answer": _single_df("format", 0.94, 0.9),
    }


def _code_metrics() -> Dict[str, pd.DataFrame]:
    return {
        "code_generate": _single_df("code", 0.66, 1.2),
        "sc_ensemble": _single_df("sc", 0.92, 0.8),
        "reflection_test": _single_df("fix", 0.41, 2.4),
    }


def test_math_auto_backward_parity_all_structures():
    workflow = get_workflow_module("math")
    metrics = _math_metrics()
    p_programmer = metrics["programmer"]["accuracy"].iloc[0]
    p_refine = metrics["refine_solver"]["accuracy"].iloc[0]
    p_detailed = metrics["detailed_solver"]["accuracy"].iloc[0]
    p_generate = metrics["generate_solver"]["accuracy"].iloc[0]
    p_sc = metrics["sc_ensemble"]["accuracy"].iloc[0]
    l_programmer = metrics["programmer"]["latency"].iloc[0]
    l_refine = metrics["refine_solver"]["latency"].iloc[0]
    l_detailed = metrics["detailed_solver"]["latency"].iloc[0]
    l_generate = metrics["generate_solver"]["latency"].iloc[0]
    l_sc = metrics["sc_ensemble"]["latency"].iloc[0]

    for structure in workflow.enumerate_structures():
        counts = structure.get("active_agent_counts", {})
        n_prog = int(counts.get("programmer", 0))
        n_refine = int(counts.get("refine_solver", 0))
        n_detailed = int(counts.get("detailed_solver", 0))
        n_generate = int(counts.get("generate_solver", 0))
        n_sc = int(counts.get("sc_ensemble", 0))

        expected_accuracy = _math_expected_accuracy(
            p_programmer=p_programmer,
            p_refine=p_refine,
            p_detailed=p_detailed,
            p_generate=p_generate,
            p_sc_ensemble=p_sc,
            n_programmer_refine=min(n_prog, n_refine),
            n_detailed=n_detailed,
            n_generate=n_generate,
            use_sc_ensemble=(n_sc > 0),
        )
        expected_latency = (
            l_programmer * n_prog
            + l_refine * n_refine
            + l_detailed * n_detailed
            + l_generate * n_generate
            + l_sc * n_sc
        )

        df = workflow.backward({"structure": structure, "metrics": metrics})
        assert len(df) == 1
        assert float(df["workflow_accuracy"].iloc[0]) == pytest.approx(float(expected_accuracy), abs=1e-10)
        assert float(df["workflow_latency"].iloc[0]) == pytest.approx(float(expected_latency), abs=1e-10)


def test_hotpot_auto_backward_parity_all_structures():
    workflow = get_workflow_module("hotpotqa")
    metrics = _hotpot_metrics()
    p_answer = metrics["answer_generate"]["accuracy"].iloc[0]
    p_sc = metrics["sc_ensemble"]["accuracy"].iloc[0]
    p_format = metrics["format_answer"]["accuracy"].iloc[0]
    l_answer = metrics["answer_generate"]["latency"].iloc[0]
    l_sc = metrics["sc_ensemble"]["latency"].iloc[0]
    l_format = metrics["format_answer"]["latency"].iloc[0]

    for structure in workflow.enumerate_structures():
        counts = structure.get("active_agent_counts", {})
        n_answer = int(counts.get("answer_generate", 0))
        n_sc = int(counts.get("sc_ensemble", 0))
        n_format = int(counts.get("format_answer", 0))

        expected_accuracy = _hotpot_expected_accuracy(
            p_answer_generate=p_answer,
            p_sc_ensemble=p_sc,
            p_format_answer=p_format,
            n_answer_generate=n_answer,
            use_sc_ensemble=(n_sc > 0),
            use_format_answer=(n_format > 0),
        )
        expected_latency = l_answer * n_answer + l_sc * n_sc + l_format * n_format

        df = workflow.backward({"structure": structure, "metrics": metrics})
        assert len(df) == 1
        assert float(df["workflow_accuracy"].iloc[0]) == pytest.approx(float(expected_accuracy), abs=1e-10)
        assert float(df["workflow_latency"].iloc[0]) == pytest.approx(float(expected_latency), abs=1e-10)


def test_livecodebench_auto_backward_parity_all_structures():
    workflow = get_workflow_module("livecodebench")
    metrics = _code_metrics()
    p_code = metrics["code_generate"]["accuracy"].iloc[0]
    p_sc = metrics["sc_ensemble"]["accuracy"].iloc[0]
    p_fix = metrics["reflection_test"]["accuracy"].iloc[0]
    l_code = metrics["code_generate"]["latency"].iloc[0]
    l_sc = metrics["sc_ensemble"]["latency"].iloc[0]
    l_fix = metrics["reflection_test"]["latency"].iloc[0]

    for structure in workflow.enumerate_structures():
        counts = structure.get("active_agent_counts", {})
        n_code = int(counts.get("code_generate", 0))
        n_sc = int(counts.get("sc_ensemble", 0))
        n_fix = int(counts.get("reflection_test", 0))
        use_sc = n_sc > 0

        if use_sc:
            p_initial = (1 - (1 - p_code) ** n_code) * p_sc
        else:
            p_initial = p_code

        expected_accuracy = _code_expected_accuracy(
            p_code_generate=p_code,
            p_sc_ensemble=p_sc,
            p_fix_code=p_fix,
            n_code_generate=n_code,
            use_sc_ensemble=use_sc,
            max_test_attempts=n_fix,
        )
        expected_latency = l_code * n_code + l_sc * n_sc
        if n_fix > 0:
            expected_latency = expected_latency + _expected_fix_attempts(
                p_initial_correct=p_initial,
                p_fix_code=p_fix,
                max_attempts=n_fix,
            ) * l_fix

        df = workflow.backward({"structure": structure, "metrics": metrics})
        assert len(df) == 1
        assert float(df["workflow_accuracy"].iloc[0]) == pytest.approx(float(expected_accuracy), abs=1e-10)
        assert float(df["workflow_latency"].iloc[0]) == pytest.approx(float(expected_latency), abs=1e-10)


def test_execution_mode_rejects_unknown_mode():
    class _SimpleWorkflow(WorkflowModule):
        workflow_type = "simple"

        def __init__(self):
            super().__init__(name="simple_dsl", execution_mode="critical_path")
            self.solver = AgentNode("solver")

        def forward(self, query):
            solution = self.solver(problem=query["problem"])
            return {
                "final_answer": solution,
                "full_solution": solution,
                "final_solution": solution,
            }

    with pytest.raises(ValueError, match="Unsupported execution_mode"):
        _SimpleWorkflow()


def test_manual_backward_override_takes_precedence():
    class _ManualWorkflow(WorkflowModule):
        workflow_type = "manual"

        def __init__(self):
            super().__init__(name="manual_dsl", execution_mode="sequential")
            self.solver = AgentNode("solver")

        def forward(self, query):
            solution = self.solver(problem=query["problem"])
            return {
                "final_answer": solution,
                "full_solution": solution,
                "final_solution": solution,
            }

        def backward(self, payload):
            ctx = self.metric_context(payload)
            return ctx.finish(workflow_accuracy=0.123, workflow_latency=4.56)

    workflow = _ManualWorkflow()
    structure = workflow.get_full_structure()
    metrics = {"solver": _single_df("solver", 0.9, 1.0)}

    df = workflow.backward({"structure": structure, "metrics": metrics})
    assert float(df["workflow_accuracy"].iloc[0]) == pytest.approx(0.123)
    assert float(df["workflow_latency"].iloc[0]) == pytest.approx(4.56)


def test_unsupported_conditional_requires_manual_backward():
    class _UnsupportedConditionalWorkflow(WorkflowModule):
        workflow_type = "unsupported_cond"

        def __init__(self):
            super().__init__(name="unsupported_cond_dsl", execution_mode="sequential")
            self.solver = AgentNode("solver")

        def forward(self, query):
            out = self.solver(problem=query["problem"])
            for _ in range(1):
                if out["status"] == "ok":
                    break
            return {
                "final_answer": out,
                "full_solution": out,
                "final_solution": out,
            }

    workflow = _UnsupportedConditionalWorkflow()
    structure = workflow.get_full_structure()
    metrics = {"solver": _single_df("solver", 0.9, 1.0)}

    with pytest.raises(ValueError, match="Auto backward currently supports only captured loop-break"):
        workflow.backward({"structure": structure, "metrics": metrics})


def test_legacy_formula_helpers_are_not_exported():
    code_pkg = importlib.import_module("flowcompile.workflows.code")
    math_pkg = importlib.import_module("flowcompile.workflows.math")
    hotpot_pkg = importlib.import_module("flowcompile.workflows.hotpotqa")

    assert not hasattr(code_pkg, "calculate_code_workflow_accuracy")
    assert not hasattr(code_pkg, "calculate_expected_fix_attempts")
    assert not hasattr(math_pkg, "calculate_math_workflow_accuracy")
    assert not hasattr(hotpot_pkg, "calculate_hotpotqa_workflow_accuracy")
