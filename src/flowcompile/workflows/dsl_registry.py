"""Registry for Python DSL workflow modules."""
from __future__ import annotations

from flowcompile.dsl.torchlike import WorkflowModule
from flowcompile.workflows.math.workflow import MathWorkflowDSL
from flowcompile.workflows.hotpotqa.workflow import HotpotQAWorkflowDSL
from flowcompile.workflows.code.workflow import LiveCodeBenchWorkflowDSL


def get_workflow_module(workflow_type: str) -> WorkflowModule:
    workflow_type = workflow_type.lower()
    if workflow_type in ("math", "gsm8k"):
        return MathWorkflowDSL()
    if workflow_type == "hotpotqa":
        return HotpotQAWorkflowDSL()
    if workflow_type == "livecodebench":
        return LiveCodeBenchWorkflowDSL()
    raise ValueError(f"Unsupported workflow_type for DSL: {workflow_type}")


__all__ = ["get_workflow_module"]
