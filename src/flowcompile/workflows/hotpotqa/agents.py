"""HotpotQA workflow agents."""
from __future__ import annotations

from flowcompile.core.llm.client import AsyncLLM
from flowcompile.core.workflow.operators import AnswerGenerate, Custom
from flowcompile.core.workflow.agents import SubAgent, AgentResult
from flowcompile.core.workflow.agents import EnsembleAgent


class AnswerGenerateAgent(SubAgent):
    """Agent wrapper for AnswerGenerate operator (used in HotpotQA workflow)."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="answer_generate",
            llm=llm,
            description="Generates an answer to a question given context",
        )
        self.operator = AnswerGenerate(llm)

    async def run(self, problem: str, **kwargs) -> AgentResult:
        response = await self.operator(problem)
        output = response.get("answer", response.get("response", str(response)))
        self.capture_operator_result(
            response,
        )
        return AgentResult(output=output)


class FormatAnswerAgent(SubAgent):
    """Agent wrapper for Custom (FormatAnswer) operator (used in HotpotQA workflow)."""

    def __init__(self, llm: AsyncLLM):
        super().__init__(
            name="format_answer",
            llm=llm,
            description="Formats and cleans the final answer",
        )
        self.custom_op = Custom(llm, "FormatAnswer")

    async def run(self, question: str, best_answer: str, instruction: str, **kwargs) -> AgentResult:
        input_text = f"Question: {question}\nBest answer: {best_answer}"
        result, input_tokens, output_tokens = await self.custom_op(
            input=input_text,
            instruction=instruction,
            return_io_tokens=True,
        )
        output = result.get("response", str(result))
        self.capture_operator_result(
            result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return AgentResult(output=output)


__all__ = ["AnswerGenerateAgent", "FormatAnswerAgent", "EnsembleAgent"]
