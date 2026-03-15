"""HotpotQA Python DSL workflow definition."""
from __future__ import annotations

from typing import Any, Dict

from workflow_compiler.dsl.torchlike import WorkflowModule, AgentNode
from workflow_compiler.core.workflow import prompts as prompt_custom
from workflow_compiler.workflows.hotpotqa.judges import get_profiling_judges


class HotpotQAWorkflowDSL(WorkflowModule):
    workflow_type = "hotpotqa"

    def __init__(self):
        super().__init__(name="hotpotqa_dsl")
        self.answer_generate = AgentNode("answer_generate")
        self.sc_ensemble = AgentNode("sc_ensemble", min_input_branches=2)
        self.format_answer = AgentNode("format_answer")

    def forward(self, query: Dict[str, Any]):
        problem = query["problem"]
        solutions = [self.answer_generate(problem=problem) for _ in range(3)]
        best = self.sc_ensemble(problem=problem, solutions=solutions)
        final = self.format_answer(
            question=problem,
            best_answer=best,
            instruction=prompt_custom.FORMAT_ANSWER_PROMPT,
        )
        return {
            "final_answer": final,
            "full_solution": final,
            "final_solution": final,
        }

    def get_profiling_judges(self):
        return get_profiling_judges()


__all__ = [
    "HotpotQAWorkflowDSL",
]
