"""Workflow-owned profiling judges for math-style workflows."""
from __future__ import annotations

from workflow_compiler.compiler.judge_types import JudgeContext, JudgeResult, WorkflowJudgeRegistry
from workflow_compiler.core.llm.formatter import XmlFormatter
from workflow_compiler.core.workflow.operators import ScEnsembleOp

_MATH_SOLUTION_PROMPT = """You are evaluating a mathematical solution.

Ground Truth:
{ground_truth}

Model Output:
{model_output}

Task: Compare the final answers and ignore minor formatting differences.

Respond with ONLY ONE WORD:
- "CORRECT" if final answers match
- "INCORRECT" otherwise
"""

_PROGRAMMER_PROMPT = """You are evaluating whether a Python code's execution output is correct.

Expected Output (from ground truth):
{ground_truth}

Actual Execution Output:
{exec_output}

Task: Compare the actual output with the expected output. Consider numerical equivalence and ignore minor formatting differences.

Respond with ONLY ONE WORD:
- "CORRECT" if outputs match
- "INCORRECT" otherwise
"""

_SC_ENSEMBLE_PROMPT = """You are evaluating an ensemble agent's solution selection for a math problem.

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


async def _judge_math_solution(evaluator, context: JudgeContext) -> JudgeResult:
    prompt = _MATH_SOLUTION_PROMPT.format(
        ground_truth=context.ground_truth,
        model_output=context.model_output,
    )
    return JudgeResult(is_correct=await evaluator.judge_with_prompt(context.agent_name, prompt))


async def judge_programmer(evaluator, context: JudgeContext) -> JudgeResult:
    model_code = evaluator.extract_code_from_output(context.model_output)
    if not model_code:
        print("Warning: Could not extract code from model output")
        return JudgeResult(is_correct=False)

    status, output = await evaluator.execute_code_in_subprocess(model_code)
    if status is None:
        print(output)
        return JudgeResult(is_correct=False)
    if status != "Success":
        print(f"Code execution failed: {output[:200]}")
        return JudgeResult(is_correct=False)

    prompt = _PROGRAMMER_PROMPT.format(
        ground_truth=context.ground_truth,
        exec_status=status,
        exec_output=output,
    )
    return JudgeResult(is_correct=await evaluator.judge_with_prompt(context.agent_name, prompt))


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
        ground_truth_solution=ground_truth_solution,
        predicted_solution=predicted_solution,
        model_output=context.model_output,
        ground_truth_output=context.ground_truth,
    )
    return JudgeResult(is_correct=await evaluator.judge_with_prompt(context.agent_name, prompt))


def get_profiling_judges() -> WorkflowJudgeRegistry:
    return {
        "programmer": judge_programmer,
        "detailed_solver": _judge_math_solution,
        "generate_solver": _judge_math_solution,
        "refine_solver": _judge_math_solution,
        "sc_ensemble": judge_sc_ensemble,
    }


__all__ = ["get_profiling_judges"]
