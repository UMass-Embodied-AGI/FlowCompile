"""Shared types for workflow-owned profiling judges."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from workflow_compiler.compiler.profiling import JudgeEvaluator


@dataclass(frozen=True)
class JudgeContext:
    """Inputs available to a workflow-owned profiling judge."""

    agent_name: str
    ground_truth: str
    model_output: str
    input_prompt: Optional[str] = None
    workflow_type: Optional[str] = None
    sample_data: Optional[Dict[str, Any]] = None
    problem: Optional[str] = None
    solutions: Optional[List[Any]] = None
    question: Optional[str] = None
    original_sample: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class JudgeResult:
    """Outcome returned by a workflow-owned profiling judge."""

    is_correct: Union[bool, float]
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None


WorkflowJudge = Callable[["JudgeEvaluator", JudgeContext], Awaitable[JudgeResult]]

WorkflowJudgeRegistry = Dict[str, WorkflowJudge]


__all__ = [
    "JudgeContext",
    "JudgeResult",
    "WorkflowJudge",
    "WorkflowJudgeRegistry",
]
