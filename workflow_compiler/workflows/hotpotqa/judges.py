"""Workflow-owned profiling judges for HotpotQA."""
from __future__ import annotations

from workflow_compiler.compiler.judge_types import JudgeContext, JudgeResult, WorkflowJudgeRegistry
from workflow_compiler.core.llm.formatter import XmlFormatter
from workflow_compiler.core.workflow.operators import AnswerGenerateOp, ScEnsembleOp

_ANSWER_GENERATE_PROMPT = """You are evaluating a question answering output.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Model Output:
{model_output}

Task: Compare the answer in the model output with the ground truth. Consider semantic equivalence and ignore minor formatting differences.

Respond with ONLY ONE WORD:
- "CORRECT" if answers are semantically equivalent
- "INCORRECT" otherwise
"""

_SC_ENSEMBLE_PROMPT = """You are evaluating an ensemble agent's answer selection for a question answering task.

Question:
{question}

Ground Truth Answer:
{ground_truth_solution}

Predicted Answer:
{predicted_solution}

Ground Truth Reasoning:
{ground_truth_output}

Agent's Reasoning:
{model_output}

Respond with ONLY ONE WORD:
- "CORRECT" if the predicted answer and reasoning are aligned with the ground truth
- "INCORRECT" otherwise
"""


def _extract_question(context: JudgeContext) -> str:
    if isinstance(context.question, str) and context.question.strip():
        return context.question.strip()
    if isinstance(context.problem, str) and context.problem.strip():
        if "Question:" in context.problem:
            return context.problem.split("Question:")[-1].split("Answer:")[0].strip()
        return context.problem.strip()
    return "N/A"


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


async def judge_answer_generate(evaluator, context: JudgeContext) -> JudgeResult:
    formatter = XmlFormatter.from_model(AnswerGenerateOp)
    is_valid_format, parsed_data = formatter.validate_response(context.model_output)
    if not is_valid_format or not parsed_data:
        return JudgeResult(is_correct=False)
    answer = parsed_data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return JudgeResult(is_correct=False)

    prompt = _ANSWER_GENERATE_PROMPT.format(
        question=_extract_question(context),
        ground_truth=context.ground_truth.strip(),
        model_output=answer.strip(),
    )
    return JudgeResult(is_correct=await evaluator.judge_with_prompt(context.agent_name, prompt))


async def judge_format_answer(evaluator, context: JudgeContext) -> JudgeResult:
    if not context.model_output or not context.model_output.strip():
        return JudgeResult(is_correct=False, metric_name="f1_score", metric_value=0.0)
    f1_score = evaluator.calculate_f1_score(context.ground_truth, context.model_output)
    return JudgeResult(is_correct=f1_score, metric_name="f1_score", metric_value=f1_score)


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
        question=_extract_question(context),
        ground_truth_solution=ground_truth_solution,
        predicted_solution=predicted_solution,
        model_output=context.model_output,
        ground_truth_output=context.ground_truth,
    )
    return JudgeResult(is_correct=await evaluator.judge_with_prompt(context.agent_name, prompt))


def get_profiling_judges() -> WorkflowJudgeRegistry:
    return {
        "answer_generate": judge_answer_generate,
        "format_answer": judge_format_answer,
        "sc_ensemble": judge_sc_ensemble,
    }


__all__ = ["get_profiling_judges"]
