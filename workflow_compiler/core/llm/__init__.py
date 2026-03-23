"""LLM-related core modules."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AsyncLLM",
    "ClientLLMConfig",
    "LLMsConfig",
    "TokenUsageTracker",
    "create_llm_instance",
    "BaseFormatter",
    "CodeFormatter",
    "FormatError",
    "TextFormatter",
    "XmlFormatter",
    "ExperimentConfig",
    "LLMConfig",
    "ThinkingBudgetLLM",
    "build_setting",
    "create_experiment_config",
    "get_config_path",
    "load_config",
    "load_model_alias_to_hf_name_map",
    "merge_configs",
    "parse_config",
    "parse_llm_config",
    "validate_config",
]


_EXPORTS = {
    "AsyncLLM": ("workflow_compiler.core.llm.client", "AsyncLLM"),
    "ClientLLMConfig": ("workflow_compiler.core.llm.client", "LLMConfig"),
    "LLMsConfig": ("workflow_compiler.core.llm.client", "LLMsConfig"),
    "TokenUsageTracker": ("workflow_compiler.core.llm.client", "TokenUsageTracker"),
    "create_llm_instance": ("workflow_compiler.core.llm.client", "create_llm_instance"),
    "BaseFormatter": ("workflow_compiler.core.llm.formatter", "BaseFormatter"),
    "CodeFormatter": ("workflow_compiler.core.llm.formatter", "CodeFormatter"),
    "FormatError": ("workflow_compiler.core.llm.formatter", "FormatError"),
    "TextFormatter": ("workflow_compiler.core.llm.formatter", "TextFormatter"),
    "XmlFormatter": ("workflow_compiler.core.llm.formatter", "XmlFormatter"),
    "ExperimentConfig": ("workflow_compiler.core.llm.config", "ExperimentConfig"),
    "LLMConfig": ("workflow_compiler.core.llm.config", "LLMConfig"),
    "ThinkingBudgetLLM": ("workflow_compiler.core.llm.config", "ThinkingBudgetLLM"),
    "build_setting": ("workflow_compiler.core.llm.config", "build_setting"),
    "create_experiment_config": ("workflow_compiler.core.llm.config", "create_experiment_config"),
    "get_config_path": ("workflow_compiler.core.llm.config", "get_config_path"),
    "load_config": ("workflow_compiler.core.llm.config", "load_config"),
    "load_model_alias_to_hf_name_map": ("workflow_compiler.core.llm.config", "load_model_alias_to_hf_name_map"),
    "merge_configs": ("workflow_compiler.core.llm.config", "merge_configs"),
    "parse_config": ("workflow_compiler.core.llm.config", "parse_config"),
    "parse_llm_config": ("workflow_compiler.core.llm.config", "parse_llm_config"),
    "validate_config": ("workflow_compiler.core.llm.config", "validate_config"),
}


def __getattr__(name: str) -> Any:
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
