"""Math Python DSL workflow definition."""
from __future__ import annotations

from typing import Any, Dict

from workflow_compiler.dsl.torchlike import WorkflowModule, AgentNode, ToolNode


class MathWorkflowDSL(WorkflowModule):
    workflow_type = "math"

    def __init__(self):
        super().__init__(name="math_dsl")
        self.programmer = AgentNode("programmer")
        self.refine_solver = AgentNode("refine_solver")
        self.detailed_solver = AgentNode("detailed_solver")
        self.generate_solver = AgentNode("generate_solver")
        self.sc_ensemble = AgentNode("sc_ensemble")
        self.extract_answer = ToolNode("extract_answer", impl="extract_math_answer")

    def forward(self, query: Dict[str, Any]):
        problem = query["problem"]
        prog = self.programmer(problem=problem)
        refined = self.refine_solver(problem=problem, context=prog)
        detailed = self.detailed_solver(problem=problem)
        solutions = [refined, detailed]
        for _ in range(2):
            solutions.append(self.generate_solver(problem=problem))
        best = self.sc_ensemble(problem=problem, solutions=solutions)
        answer = self.extract_answer(solution=best)
        return {
            "final_answer": answer,
            "full_solution": best,
            "final_solution": best,
        }


__all__ = [
    "MathWorkflowDSL",
]
