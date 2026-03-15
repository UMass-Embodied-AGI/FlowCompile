"""Workflow-owned profiling judges for LiveCodeBench."""
from __future__ import annotations

from workflow_compiler.compiler.judge_types import JudgeContext, JudgeResult, WorkflowJudgeRegistry
from workflow_compiler.core.llm.formatter import XmlFormatter
from workflow_compiler.core.workflow.operators import ScEnsembleOp

_SC_ENSEMBLE_PROMPT = """You are evaluating an ensemble agent's code solution selection for a coding problem.

Problem:
{problem}

Ground Truth Solution:
{ground_truth_solution}

Predicted Solution:
{predicted_solution}

Ground Truth Reasoning:
{ground_truth_output}

Agent's Reasoning:
{model_output}

Respond with ONLY ONE WORD:
- "CORRECT" if the predicted solution and reasoning are aligned with the ground truth
- "INCORRECT" otherwise
"""


def _extract_ground_truth_solution_letter(evaluator, ground_truth: str) -> str | None:
    formatter = XmlFormatter.from_model(ScEnsembleOp)
    try:
        _ok, parsed_data = formatter.validate_response(ground_truth)
        if isinstance(parsed_data, dict):
            candidate = parsed_data.get("solution_letter", "")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().upper()
    except Exception:
        pass
    return evaluator.extract_boxed_choice_letter(ground_truth)


async def _judge_private_tests(evaluator, context: JudgeContext) -> JudgeResult:
    model_code = evaluator.extract_code_from_output(context.model_output)
    if model_code is None:
        model_code = context.model_output
    if not model_code or not model_code.strip():
        print(f"Warning: Empty code output for {context.agent_name}")
        return JudgeResult(is_correct=False)
    if context.original_sample is None:
        print(f"Warning: No original_sample provided for private test evaluation of {context.agent_name}")
        return JudgeResult(is_correct=False)
    is_correct = await evaluator.evaluate_code_with_private_tests(model_code, context.original_sample)
    return JudgeResult(is_correct=is_correct)


async def judge_sc_ensemble(evaluator, context: JudgeContext) -> JudgeResult:
    predicted_solution_letter = evaluator.extract_boxed_choice_letter(context.model_output)
    if not predicted_solution_letter:
        return JudgeResult(is_correct=False)

    ground_truth_solution_letter = _extract_ground_truth_solution_letter(
        evaluator,
        context.ground_truth,
    )
    if not ground_truth_solution_letter:
        return JudgeResult(is_correct=False)

    if not isinstance(context.solutions, list) or not context.solutions:
        return JudgeResult(is_correct=predicted_solution_letter == ground_truth_solution_letter)

    gt_idx = ord(ground_truth_solution_letter) - ord("A")
    pred_idx = ord(predicted_solution_letter) - ord("A")
    if gt_idx < 0 or pred_idx < 0 or gt_idx >= len(context.solutions) or pred_idx >= len(context.solutions):
        return JudgeResult(is_correct=predicted_solution_letter == ground_truth_solution_letter)

    ground_truth_solution = context.solutions[gt_idx]
    try:
        predicted_solution = context.solutions[pred_idx]
    except Exception:
        return JudgeResult(is_correct=predicted_solution_letter == ground_truth_solution_letter)

    prompt = _SC_ENSEMBLE_PROMPT.format(
        problem=context.problem or "N/A",
        ground_truth_solution=ground_truth_solution,
        predicted_solution=predicted_solution,
        model_output=context.model_output,
        ground_truth_output=context.ground_truth,
    )
    return JudgeResult(is_correct=await evaluator.judge_with_prompt(context.agent_name, prompt))


def get_profiling_judges() -> WorkflowJudgeRegistry:
    return {
        "code_generate": _judge_private_tests,
        "reflection_test": _judge_private_tests,
        "sc_ensemble": judge_sc_ensemble,
    }


__all__ = ["get_profiling_judges"]
