"""LiveCodeBench Python DSL workflow definition."""
from __future__ import annotations

from typing import Any, Dict

from workflow_compiler.dsl.torchlike import WorkflowModule, AgentNode, ToolNode
from workflow_compiler.core.workflow import prompts as prompt_custom


class LiveCodeBenchWorkflowDSL(WorkflowModule):
    workflow_type = "livecodebench"

    def __init__(self):
        super().__init__(name="livecodebench_dsl")
        self.code_generate = AgentNode("code_generate")
        self.sc_ensemble = AgentNode("sc_ensemble")
        self.test = ToolNode("test", impl="run_code_tests")
        self.reflection_test = AgentNode("reflection_test")

    def forward(self, query: Dict[str, Any]):
        problem = query["problem"]
        entry_point = query["entry_point"]
        question_id = query["question_id"]
        dataset_name = "LiveCodeBench"
        solutions = [
            self.code_generate(
                problem=problem,
                entry_point=entry_point,
                instruction=prompt_custom.CODE_GENERATE_PROMPT,
            )
            for _ in range(3)
        ]
        best = self.sc_ensemble(problem=problem, solutions=solutions)
        current = best
        for _ in range(3):
            test_out = self.test(
                problem=problem,
                solution=current,
                entry_point=entry_point,
                dataset=dataset_name,
                question_id=question_id,
            )
            if test_out["test_passed"]:
                break
            current = self.reflection_test(
                problem=problem,
                solution=current,
                error=test_out["error"],
                error_type=test_out["error_type"],
                entry_point=entry_point,
            )
        return {
            "final_answer": current,
            "full_solution": current,
            "final_solution": current,
        }


__all__ = [
    "LiveCodeBenchWorkflowDSL",
]
