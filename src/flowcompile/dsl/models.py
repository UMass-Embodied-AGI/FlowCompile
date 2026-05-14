"""Pydantic models for FlowCompile DSL."""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_budget: Optional[Any] = None


class NodeSpec(BaseModel):
    id: str
    type: str = Field(..., description="agent|tool|router|ensemble|formatter")
    name: str
    llm_ref: Optional[str] = None
    impl: Optional[str] = None
    llm: Optional[LLMConfig] = None
    prompt: Optional[str] = None
    prompt_ref: Optional[str] = None
    io: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _check_prompt(self):
        if self.type == "agent":
            if self.llm is None and self.llm_ref is None:
                raise ValueError("agent nodes must define llm or llm_ref settings")
        return self


class EdgeSpec(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    when: Optional[Dict[str, Any]] = None
    max_visits: Optional[int] = None


class WorkflowSpec(BaseModel):
    version: str
    name: str
    entry: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]
