"""FlowCompile compiler package."""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "compile_pareto",
    "run_latency_benchmark",
    "run_ground_truth",
    "run_agent_dataset",
    "run_profiling",
    "run_validation",
]

_LAZY_EXPORTS = {
    "compile_pareto": ("workflow_compiler.compiler.pipeline", "compile_pareto"),
    "run_latency_benchmark": ("workflow_compiler.compiler.latency", "run_latency_benchmark"),
    "run_ground_truth": ("workflow_compiler.compiler.ground_truth", "run_ground_truth"),
    "run_agent_dataset": ("workflow_compiler.compiler.agent_dataset", "run_agent_dataset"),
    "run_profiling": ("workflow_compiler.compiler.profiling", "run_profiling"),
    "run_validation": ("workflow_compiler.compiler.validation", "run_validation"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
