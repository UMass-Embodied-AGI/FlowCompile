"""Base agent definitions shared across workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Tuple

from workflow_compiler.core.llm.client import AsyncLLM
from workflow_compiler.core.logs import logger
from workflow_compiler.core.workflow.operators import ScEnsemble


@dataclass
class AgentResult:
    """Container returned by SubAgent.run."""

    output: Any
    status: str = "success"
    extras: Dict[str, Any] = field(default_factory=dict)


class SubAgent:
    """Base class for all sub-agents with automatic metadata recording."""

    _RESERVED_META_KEYS = {
        "agent",
        "call_number",
        "input_tokens",
        "output_tokens",
        "raw_llm_prompt",
        "raw_llm_output",
        "processed_output",
        "timestamp",
        "status",
    }

    def __init__(self, name: str, llm: AsyncLLM, description: str, requires_llm_capture: bool = True):
        """
        Initialize a sub-agent.

        Args:
            name: Unique identifier for the agent
            llm: Language model instance for this agent
            description: Brief description of what this agent does
            requires_llm_capture: Whether this agent must call capture helpers
        """
        self.name = name
        self.llm = llm
        self.description = description
        self.call_count = 0
        self.requires_llm_capture = requires_llm_capture
        self._reset_capture_state()

    def _reset_capture_state(self) -> None:
        self._captured_input_tokens = 0
        self._captured_output_tokens = 0
        self._captured_raw_prompt = ""
        self._captured_raw_output = ""
        self._captured_any_llm_call = False

    def capture_llm_interaction(
        self,
        *,
        raw_prompt: str = "",
        raw_output: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record one LLM interaction. Subclasses should use this helper."""
        self._captured_any_llm_call = True
        self._captured_input_tokens += int(input_tokens or 0)
        self._captured_output_tokens += int(output_tokens or 0)
        self._captured_raw_prompt = str(raw_prompt or "")
        self._captured_raw_output = str(raw_output or "")

    def capture_operator_result(
        self,
        result: Any,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Capture metadata from operator responses that include _raw/_token fields."""
        prompt = ""
        output = ""
        if isinstance(result, dict):
            prompt = result.get("_raw_llm_prompt", "")
            output = result.get("_raw_llm_output", "")
            input_tokens = result.get("_input_tokens", input_tokens)
            output_tokens = result.get("_output_tokens", output_tokens)
        self.capture_llm_interaction(
            raw_prompt=str(prompt or ""),
            raw_output=str(output or ""),
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
        )

    async def run(self, **kwargs) -> AgentResult:
        """
        Implement the core agent logic.

        Args:
            **kwargs: Agent input arguments from workflow node

        Returns:
            AgentResult containing processed output and optional extra metadata
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _build_metadata(
        self,
        *,
        call_number: int,
        status: str,
        processed_output: Any,
        extras: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "agent": self.name,
            "call_number": call_number,
            "input_tokens": self._captured_input_tokens,
            "output_tokens": self._captured_output_tokens,
            "raw_llm_prompt": self._captured_raw_prompt,
            "raw_llm_output": self._captured_raw_output,
            "processed_output": processed_output,
            "timestamp": datetime.now().isoformat(),
            "status": status,
        }
        for key, value in (extras or {}).items():
            if key in self._RESERVED_META_KEYS:
                logger.warning(f"Ignoring reserved metadata key '{key}' returned by agent '{self.name}'")
                continue
            metadata[key] = value
        return metadata

    async def execute(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute one agent call with automatic metadata recording.
        """
        self.call_count += 1
        call_number = self.call_count
        self._reset_capture_state()

        try:
            result = await self.run(**kwargs)
            if isinstance(result, AgentResult):
                agent_result = result
            else:
                agent_result = AgentResult(output=result)

            if (
                self.requires_llm_capture
                and agent_result.status == "success"
                and not self._captured_any_llm_call
            ):
                raise RuntimeError(
                    f"Agent '{self.name}' must use capture helpers to record raw prompt/output and token counts."
                )

            metadata = self._build_metadata(
                call_number=call_number,
                status=agent_result.status,
                processed_output=agent_result.output,
                extras=agent_result.extras,
            )
            return agent_result.output, metadata
        except Exception as exc:
            logger.error(f"{self.name} error: {exc}")
            error_output = f"Error in {self.name}: {str(exc)}"
            metadata = self._build_metadata(
                call_number=call_number,
                status="error",
                processed_output=error_output,
                extras={"error": str(exc)},
            )
            return error_output, metadata

    def get_statistics(self) -> Dict:
        """Get usage statistics for this agent."""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "usage_summary": self.llm.get_usage_summary(),
        }

class EnsembleAgent(SubAgent):
    """Ensemble agent that selects best solution from multiple candidates."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="sc_ensemble",
            llm=llm,
            description="Selects the best solution from multiple candidate solutions using self-consistency",
        )
        self.ensemble_op = ScEnsemble(llm, "ScEnsemble")

    async def run(self, problem: str = None, solutions: list = None, **kwargs) -> AgentResult:
        """Select best solution with self-consistency."""
        if problem is not None and isinstance(problem, list):
            solutions, problem = problem, solutions
        if problem is None:
            problem = kwargs.get("question", kwargs.get("problem"))
        if solutions is None:
            solutions = kwargs.get("solutions")

        if not problem or not solutions:
            raise ValueError("Both 'problem' and 'solutions' parameters are required")

        try:
            result, input_tokens, output_tokens = await self.ensemble_op(
                solutions=solutions,
                problem=problem,
                return_io_tokens=True,
            )
            best_solution = result.get("response", str(result))
            self.capture_operator_result(
                result,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            return AgentResult(
                output=best_solution,
                extras={
                    "num_solutions": len(solutions),
                    "solution_letter": result.get("_solution_letter", ""),
                    "selected_index": result.get("_selected_index", -1),
                },
            )

        except Exception as e:
            logger.error(f"Ensemble error: {e}")
            fallback_solution = solutions[0] if solutions and len(solutions) > 0 else ""
            if fallback_solution:
                logger.warning("Ensemble failed, falling back to first solution")
            else:
                logger.warning("Ensemble failed, no solutions available to fall back to")
            return AgentResult(
                output=fallback_solution,
                status="error",
                extras={
                    "error": str(e),
                    "fallback_to": "first_solution" if fallback_solution else "empty",
                },
            )


__all__ = ["SubAgent", "AgentResult", "EnsembleAgent"]
