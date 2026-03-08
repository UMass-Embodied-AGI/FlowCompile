"""Registry for Python DSL workflow modules."""
from __future__ import annotations

from workflow_compiler.dsl.torchlike import WorkflowModule
from workflow_compiler.workflows.math.workflow import MathWorkflowDSL
from workflow_compiler.workflows.hotpotqa.workflow import HotpotQAWorkflowDSL
from workflow_compiler.workflows.code.workflow import LiveCodeBenchWorkflowDSL
from workflow_compiler.workflows.openclaw_lobster.workflow import OpenClawLobsterWorkflowDSL


def get_workflow_module(workflow_type: str, **kwargs) -> WorkflowModule:
    workflow_type = workflow_type.lower()
    if workflow_type in ("math", "gsm8k"):
        return MathWorkflowDSL()
    if workflow_type == "hotpotqa":
        return HotpotQAWorkflowDSL()
    if workflow_type == "livecodebench":
        return LiveCodeBenchWorkflowDSL()
    if workflow_type == "openclaw_lobster":
        workflow_file = kwargs.get("openclaw_lobster_workflow_file") or kwargs.get("workflow_file")
        return OpenClawLobsterWorkflowDSL(workflow_file=workflow_file)
    raise ValueError(f"Unsupported workflow_type for DSL: {workflow_type}")


__all__ = ["get_workflow_module"]
