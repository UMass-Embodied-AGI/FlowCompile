"""Template profiling judges for workflow-owned evaluation logic."""
from __future__ import annotations

from workflow_compiler.compiler.judge_types import JudgeContext, JudgeResult, WorkflowJudgeRegistry


async def judge_solver(evaluator, context: JudgeContext) -> JudgeResult:
    prompt = (
        "You are evaluating a workflow agent output.\n\n"
        f"Ground Truth:\n{context.ground_truth}\n\n"
        f"Model Output:\n{context.model_output}\n\n"
        "Respond with ONLY ONE WORD:\n"
        '- "CORRECT" if the outputs are equivalent\n'
        '- "INCORRECT" otherwise'
    )
    return JudgeResult(is_correct=await evaluator.judge_with_prompt(context.agent_name, prompt))


def get_profiling_judges() -> WorkflowJudgeRegistry:
    return {
        "solver": judge_solver,
    }


__all__ = ["get_profiling_judges"]
