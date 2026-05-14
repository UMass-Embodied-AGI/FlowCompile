"""Registries for DSL execution."""
from __future__ import annotations

from typing import Callable, Dict, Any

from flowcompile.workflows.math.agents import (
    ProgrammerAgent,
    RefineSolverAgent,
    DetailedSolverAgent,
    GenerateSolverAgent,
)
from flowcompile.workflows.hotpotqa.agents import AnswerGenerateAgent, FormatAnswerAgent
from flowcompile.workflows.code.agents import CodeGenerateAgent, TestAgent, ReflectionTestAgent
from flowcompile.core.workflow.agents import EnsembleAgent
from flowcompile.core.workflow.operators import Test


def get_agent_factory(workflow_type: str, agent_name: str):
    workflow_type = workflow_type.lower()
    if workflow_type in ("math", "gsm8k"):
        mapping = {
            "programmer": ProgrammerAgent,
            "refine_solver": RefineSolverAgent,
            "detailed_solver": DetailedSolverAgent,
            "generate_solver": GenerateSolverAgent,
            "sc_ensemble": EnsembleAgent,
        }
        return mapping.get(agent_name)

    if workflow_type == "hotpotqa":
        mapping = {
            "answer_generate": AnswerGenerateAgent,
            "sc_ensemble": EnsembleAgent,
            "format_answer": FormatAnswerAgent,
        }
        return mapping.get(agent_name)

    if workflow_type in ("livecodebench", "code"):
        mapping = {
            "code_generate": CodeGenerateAgent,
            "sc_ensemble": EnsembleAgent,
            "test": TestAgent,
            "reflection": ReflectionTestAgent,
            "reflection_test": ReflectionTestAgent,
        }
        return mapping.get(agent_name)

    return None


# Tool registry

def tool_extract_math_answer(solution: str) -> str:
    import re
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    matches = re.findall(pattern, solution or "", re.DOTALL)
    if matches:
        return matches[-1].strip()
    # fallback to last sentence
    sentence_end_pattern = r"(?<!\d)[.!?]\s+"
    sentences = re.split(sentence_end_pattern, solution or "")
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[-1] if sentences else (solution or "")


def tool_ensure_nonempty(primary: list, fallback: list) -> list:
    if primary:
        return primary
    return fallback or []


def tool_first_nonempty(primary: list, fallback: list):
    def _is_nonempty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    for value in (primary or []):
        if _is_nonempty(value):
            return value
    for value in (fallback or []):
        if _is_nonempty(value):
            return value
    return None


def tool_identity(value):
    return value


_TEST_TOOL_INSTANCE: Any = None


def _get_test_tool() -> Test:
    global _TEST_TOOL_INSTANCE
    if _TEST_TOOL_INSTANCE is None:
        # Test operator does not call LLM APIs in exec_code(); llm is unused.
        _TEST_TOOL_INSTANCE = Test(llm=None, name="DslTestTool")
    return _TEST_TOOL_INSTANCE


def tool_run_code_tests(
    problem: str,
    solution: str,
    entry_point: str,
    dataset: str = "LiveCodeBench",
    question_id: str = "",
) -> Dict[str, Any]:
    del problem
    result = _get_test_tool().exec_code(
        solution=solution,
        entry_point=entry_point,
        dataset=dataset,
        question_id=question_id,
    )
    if result == "no error":
        return {
            "test_passed": True,
            "final_solution": solution,
            "error": "",
            "error_type": "",
        }
    if isinstance(result, dict) and "exec_fail_case" in result:
        return {
            "test_passed": False,
            "final_solution": solution,
            "error": result.get("exec_fail_case", ""),
            "error_type": "exec_failure",
        }
    return {
        "test_passed": False,
        "final_solution": solution,
        "error": result,
        "error_type": "test_failure",
    }


TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "extract_math_answer": tool_extract_math_answer,
    "ensure_nonempty": tool_ensure_nonempty,
    "first_nonempty": tool_first_nonempty,
    "identity": tool_identity,
    "run_code_tests": tool_run_code_tests,
}


def get_tool(impl: str):
    return TOOL_REGISTRY.get(impl)
