"""OpenClaw Lobster workflow adapter for FlowCompile DSL interfaces."""
from __future__ import annotations

import copy
from typing import Any, Dict, List

from workflow_compiler.dsl.torchlike import WorkflowModule
from workflow_compiler.workflows.openclaw_lobster.parser import parse_lobster_workflow


class OpenClawLobsterWorkflowDSL(WorkflowModule):
    workflow_type = "openclaw_lobster"

    def __init__(self, workflow_file: str):
        super().__init__(name="openclaw_lobster_dsl")
        if not workflow_file:
            raise ValueError("openclaw_lobster workflow requires 'openclaw_lobster_workflow_file'.")
        self.workflow_file = str(workflow_file)

    def forward(self, query: Dict[str, Any]):
        del query
        raise RuntimeError("OpenClaw Lobster workflow uses parsed YAML specs and is not executable via forward().")

    def compile(self) -> Dict[str, Any]:
        return parse_lobster_workflow(self.workflow_file)

    def enumerate_structures(self) -> List[Dict[str, Any]]:
        if self._structures_cache is None:
            spec = self._compile_cached()
            agent_nodes = [node for node in (spec.get("nodes") or []) if node.get("type") == "agent"]
            active_node_ids = [str(node.get("id")) for node in agent_nodes if node.get("id")]
            active_agent_counts = {str(node.get("name")): 1 for node in agent_nodes if node.get("name")}

            total_branches = 1
            for node in agent_nodes:
                metadata = node.get("metadata") or {}
                if str(metadata.get("operator") or "").lower() != "map_reduce":
                    continue
                io = node.get("io") or {}
                inputs = io.get("inputs") if isinstance(io, dict) else {}
                items = inputs.get("items") if isinstance(inputs, dict) else None
                if isinstance(items, list):
                    total_branches = max(total_branches, len(items))

            structure = {
                "structure_id": "openclaw_lobster_full",
                "active_agent_counts": active_agent_counts,
                "active_node_ids": active_node_ids,
                "remove_node_ids": [],
                "total_branches": int(total_branches),
                "is_full": True,
            }
            self._structures_cache = [structure]

        return [copy.deepcopy(item) for item in self._structures_cache]


__all__ = ["OpenClawLobsterWorkflowDSL"]
