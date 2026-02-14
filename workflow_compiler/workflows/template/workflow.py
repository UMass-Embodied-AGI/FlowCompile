"""Template Python DSL workflow definition.

Copy this file into a new workflow folder and customize it.
"""
from __future__ import annotations

from typing import Any, Dict

from workflow_compiler.dsl.torchlike import WorkflowModule, AgentNode, ToolNode


class TemplateWorkflowDSL(WorkflowModule):
    workflow_type = "template"

    def __init__(self):
        super().__init__(name="template_dsl", execution_mode="sequential")
        self.solver = AgentNode("solver")
        self.extract = ToolNode("extract", impl="extract_answer")

    def forward(self, query: Dict[str, Any]):
        problem = query["problem"]
        solution = self.solver(problem=problem)
        answer = self.extract(solution=solution)
        return {
            "final_answer": answer,
            "full_solution": solution,
            "final_solution": solution,
        }


__all__ = ["TemplateWorkflowDSL"]
