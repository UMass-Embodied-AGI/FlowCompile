"""Python builder for FlowCompile DSL."""
from __future__ import annotations

from typing import Dict, Any, Optional, List
import json

from .models import WorkflowSpec, NodeSpec, EdgeSpec, LLMConfig


class WorkflowBuilder:
    def __init__(self, name: str, version: str = "v1", description: Optional[str] = None):
        self.name = name
        self.version = version
        self.description = description
        self.nodes: List[NodeSpec] = []
        self.edges: List[EdgeSpec] = []

    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: str = "agent",
        prompt: Optional[str] = None,
        prompt_ref: Optional[str] = None,
        llm: Optional[Dict[str, Any]] = None,
        io: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        logging: Optional[Dict[str, Any]] = None,
    ):
        llm_cfg = LLMConfig(**llm) if llm else None
        node = NodeSpec(
            id=node_id,
            type=node_type,
            name=name,
            prompt=prompt,
            prompt_ref=prompt_ref,
            llm=llm_cfg,
            io=io,
            metadata=metadata,
            logging=logging,
        )
        self.nodes.append(node)
        return self

    def add_edge(self, from_node: str, to_node: str, when: Optional[Dict[str, Any]] = None, max_visits: Optional[int] = None):
        # Pydantic alias handling: create dict and re-parse to set fields correctly
        edge = EdgeSpec.model_validate({"from": from_node, "to": to_node, "when": when, "max_visits": max_visits})
        self.edges.append(edge)
        return self

    def to_spec(self) -> WorkflowSpec:
        return WorkflowSpec(
            version=self.version,
            name=self.name,
            description=self.description,
            nodes=self.nodes,
            edges=self.edges,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_spec().model_dump(by_alias=True)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
