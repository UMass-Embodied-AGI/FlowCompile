from __future__ import annotations

from workflow_compiler.dsl.torchlike import AgentNode, WorkflowModule


class _MarkerWorkflow(WorkflowModule):
    workflow_type = "marker"

    def __init__(self):
        super().__init__(name="marker_dsl")
        self.gen = AgentNode("gen")
        self.gen_alt = AgentNode("gen_alt")
        self.merge = AgentNode("merge", min_input_branches=2)

    def forward(self, query):
        first = self.gen(problem=query["problem"])
        second = self.gen_alt(problem=query["problem"])
        merged = self.merge(problem=query["problem"], solutions=[first, second])
        return {
            "final_answer": merged,
            "full_solution": merged,
            "final_solution": merged,
        }


def test_agent_node_min_input_branches_persisted_to_metadata():
    workflow = _MarkerWorkflow()
    spec = workflow.compile()
    merge_node = next(node for node in spec["nodes"] if node.get("name") == "merge")
    assert merge_node.get("metadata", {}).get("min_input_branches") == 2


def test_structure_inference_enforces_min_input_branches():
    workflow = _MarkerWorkflow()
    structures = workflow.enumerate_structures()

    saw_single_branch_without_merge = False
    for structure in structures:
        counts = structure.get("active_agent_counts") or {}
        gen_count = int(counts.get("gen", 0))
        gen_alt_count = int(counts.get("gen_alt", 0))
        merge_count = int(counts.get("merge", 0))
        upstream_branch_count = int(gen_count > 0) + int(gen_alt_count > 0)

        if merge_count > 0:
            assert upstream_branch_count >= 2
        elif upstream_branch_count == 1:
            saw_single_branch_without_merge = True

    assert saw_single_branch_without_merge
