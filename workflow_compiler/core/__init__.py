"""Core modules for FlowCompile."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AsyncLLM",
    "ClientLLMConfig",
    "LLMsConfig",
    "create_llm_instance",
    "BaseFormatter",
    "CodeFormatter",
    "FormatError",
    "TextFormatter",
    "XmlFormatter",
    "AgentResult",
    "EnsembleAgent",
    "SubAgent",
    "AnswerGenerate",
    "Custom",
    "CustomCodeGenerate",
    "MdEnsemble",
    "Programmer",
    "ReflectionTest",
    "Review",
    "Revise",
    "ScEnsemble",
    "Test",
    "run_code",
    "logger",
]


_EXPORTS = {
    "AsyncLLM": ("workflow_compiler.core.llm.client", "AsyncLLM"),
    "ClientLLMConfig": ("workflow_compiler.core.llm.client", "LLMConfig"),
    "LLMsConfig": ("workflow_compiler.core.llm.client", "LLMsConfig"),
    "create_llm_instance": ("workflow_compiler.core.llm.client", "create_llm_instance"),
    "BaseFormatter": ("workflow_compiler.core.llm.formatter", "BaseFormatter"),
    "CodeFormatter": ("workflow_compiler.core.llm.formatter", "CodeFormatter"),
    "FormatError": ("workflow_compiler.core.llm.formatter", "FormatError"),
    "TextFormatter": ("workflow_compiler.core.llm.formatter", "TextFormatter"),
    "XmlFormatter": ("workflow_compiler.core.llm.formatter", "XmlFormatter"),
    "AgentResult": ("workflow_compiler.core.workflow.agents", "AgentResult"),
    "EnsembleAgent": ("workflow_compiler.core.workflow.agents", "EnsembleAgent"),
    "SubAgent": ("workflow_compiler.core.workflow.agents", "SubAgent"),
    "AnswerGenerate": ("workflow_compiler.core.workflow.operators", "AnswerGenerate"),
    "Custom": ("workflow_compiler.core.workflow.operators", "Custom"),
    "CustomCodeGenerate": ("workflow_compiler.core.workflow.operators", "CustomCodeGenerate"),
    "MdEnsemble": ("workflow_compiler.core.workflow.operators", "MdEnsemble"),
    "Programmer": ("workflow_compiler.core.workflow.operators", "Programmer"),
    "ReflectionTest": ("workflow_compiler.core.workflow.operators", "ReflectionTest"),
    "Review": ("workflow_compiler.core.workflow.operators", "Review"),
    "Revise": ("workflow_compiler.core.workflow.operators", "Revise"),
    "ScEnsemble": ("workflow_compiler.core.workflow.operators", "ScEnsemble"),
    "Test": ("workflow_compiler.core.workflow.operators", "Test"),
    "run_code": ("workflow_compiler.core.workflow.operators", "run_code"),
    "logger": ("workflow_compiler.core.logs", "logger"),
}


def __getattr__(name: str) -> Any:
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
