"""Math workflow agents."""
from __future__ import annotations

from flowcompile.core.llm.client import AsyncLLM
from flowcompile.core.workflow.operators import Custom, Programmer
from flowcompile.core.workflow.prompts import SIMPLE_MATH_SOLVE_PROMPT
from flowcompile.core.workflow.agents import AgentResult, EnsembleAgent, SubAgent

# Import the custom prompts from the graph workflow (fallback to default prompts)
try:
    from data.results.MATH.graphs_test.round_5 import prompt as prompt_custom
except Exception:
    from flowcompile.core.workflow import prompts as prompt_custom


class GenerateSolverAgent(SubAgent):
    """Generate solver agent using GENERATE_SOLUTION_PROMPT."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="generate_solver",
            llm=llm,
            description="Generates a general solution to a math problem step by step",
        )
        self.custom_op = Custom(llm, "GenerateSolver")

    async def run(self, problem: str, **kwargs) -> AgentResult:
        result, input_tokens, output_tokens = await self.custom_op(
            input=problem,
            instruction=prompt_custom.GENERATE_SOLUTION_PROMPT,
            return_io_tokens=True,
        )
        solution = result.get("response", str(result))
        self.capture_operator_result(
            result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return AgentResult(output=solution)


class SimpleMathSolverAgent(SubAgent):
    """Simple math solver agent for GSM8K and similar grade-school math problems."""

    def __init__(self, llm: AsyncLLM, instruction: str = None):
        super().__init__(
            name="simple_math_solver",
            llm=llm,
            description="Directly solves a math problem with step-by-step reasoning",
        )
        self.custom_op = Custom(llm, "SimpleMathSolver")
        self.instruction = instruction if instruction is not None else SIMPLE_MATH_SOLVE_PROMPT

    async def run(self, problem: str, **kwargs) -> AgentResult:
        formatted_instruction = self.instruction.format(problem=problem)
        result, input_tokens, output_tokens = await self.custom_op(
            input="",
            instruction=formatted_instruction,
            return_io_tokens=True,
        )
        solution = result.get("response", str(result))
        self.capture_operator_result(
            result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return AgentResult(output=solution)


class DetailedSolverAgent(SubAgent):
    """Detailed solver agent using DETAILED_SOLUTION_PROMPT."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="detailed_solver",
            llm=llm,
            description="Provides a comprehensive, detailed step-by-step solution with mathematical rigor",
        )
        self.custom_op = Custom(llm, "DetailedSolver")

    async def run(self, problem: str, **kwargs) -> AgentResult:
        result, input_tokens, output_tokens = await self.custom_op(
            input=problem,
            instruction=prompt_custom.DETAILED_SOLUTION_PROMPT,
            return_io_tokens=True,
        )
        solution = result.get("response", str(result))
        self.capture_operator_result(
            result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return AgentResult(output=solution)


class RefineSolverAgent(SubAgent):
    """Refine solver agent using REFINE_ANSWER_PROMPT."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="refine_solver",
            llm=llm,
            description="Refines and formats a solution based on additional context such as code output",
        )
        self.custom_op = Custom(llm, "RefineSolver")

    async def run(self, problem: str, context: str, **kwargs) -> AgentResult:
        input_text = problem + f"\n{context}"
        result, input_tokens, output_tokens = await self.custom_op(
            input=input_text,
            instruction=prompt_custom.REFINE_ANSWER_PROMPT,
            return_io_tokens=True,
        )
        solution = result.get("response", str(result))
        self.capture_operator_result(
            result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return AgentResult(output=solution)


class ProgrammerAgent(SubAgent):
    """Programmer agent that generates and executes Python code."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="programmer",
            llm=llm,
            description="Generates and executes Python code to solve math problems",
        )
        self.programmer_op = Programmer(llm, "Programmer")

    async def run(self, problem: str, analysis: str = "None", **kwargs) -> AgentResult:
        result, input_tokens, output_tokens = await self.programmer_op(
            problem=problem,
            analysis=analysis,
            return_io_tokens=True,
        )
        code = result.get("code", "")
        output = result.get("output", "")
        response_text = f"Code:\n```python\n{code}\n```\n\nOutput: {output}"

        self.capture_operator_result(
            result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return AgentResult(
            output=response_text,
            extras={
                "code": code,
                "execution_output": output,
                "feedback": result.get("_feedback", ""),
            },
        )


__all__ = [
    "GenerateSolverAgent",
    "SimpleMathSolverAgent",
    "DetailedSolverAgent",
    "RefineSolverAgent",
    "ProgrammerAgent",
    "EnsembleAgent",
]
