"""Code generation workflow agents."""
from __future__ import annotations

from workflow_compiler.core.llm.client import AsyncLLM
from workflow_compiler.core.workflow.operators import CustomCodeGenerate, Test, ReflectionTest
from workflow_compiler.core.workflow.agents import SubAgent, AgentResult
from workflow_compiler.core.workflow.agents import EnsembleAgent


class CodeGenerateAgent(SubAgent):
    """Code generation agent using CustomCodeGenerate operator."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="code_generate",
            llm=llm,
            description="Generates Python code to solve a programming problem",
        )
        self.operator = CustomCodeGenerate(llm, "CodeGenerate")

    async def run(self, problem: str, entry_point: str, instruction: str, **kwargs) -> AgentResult:
        response, input_tokens, output_tokens = await self.operator(
            problem=problem,
            entry_point=entry_point,
            instruction=instruction,
            return_io_tokens=True,
        )
        code = response.get("response", str(response))
        self.capture_operator_result(
            response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return AgentResult(output=code)


class TestAgent(SubAgent):
    """Test agent using Test operator."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="test",
            llm=llm,
            description="Tests a code solution against test cases",
            requires_llm_capture=False,
        )
        self.operator = Test(llm, "Test")

    async def run(
        self,
        problem: str,
        solution: str,
        entry_point: str,
        dataset: str = "MBPP",
        question_id: str = "",
        **kwargs,
    ) -> AgentResult:
        result, _input_tokens, _output_tokens = await self.operator(
            problem=problem,
            solution=solution,
            entry_point=entry_point,
            return_io_tokens=True,
            dataset=dataset,
            question_id=question_id,
        )
        test_passed = result.get("result", False)
        final_solution = result.get("solution", solution)
        error_value = result.get("error")
        error_type = result.get("error_type")
        extras = {
            "test_passed": test_passed,
            "final_solution": final_solution,
        }
        if error_value is not None:
            extras["error"] = error_value
        if error_type is not None:
            extras["error_type"] = error_type
        return AgentResult(output=result, extras=extras)


class ReflectionTestAgent(SubAgent):
    """Reflection test agent using ReflectionTest operator."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="reflection_test",
            llm=llm,
            description="Reflects on test failures and generates an improved code solution",
        )
        self.operator = ReflectionTest(llm, "ReflectionTest")

    async def run(
        self,
        problem: str,
        solution: str,
        error: str,
        error_type: str,
        entry_point: str,
        **kwargs,
    ) -> AgentResult:
        result, input_tokens, output_tokens = await self.operator(
            problem=problem,
            solution=solution,
            error=error,
            error_type=error_type,
            entry_point=entry_point,
            return_io_tokens=True,
        )
        improved_solution = result.get("response", "")
        self.capture_operator_result(
            result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return AgentResult(
            output=improved_solution,
            extras={"error_type": error_type},
        )


__all__ = ["CodeGenerateAgent", "TestAgent", "ReflectionTestAgent", "EnsembleAgent"]
