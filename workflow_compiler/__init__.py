"""
FlowCompile: Pareto-optimal agentic workflow compilation.

This package provides:
- Core: Core components (LLM, operators, workflows, evaluators, analysis, prediction, utils)
- Benchmarks: Extensible benchmark implementations and registry
- Workflows: Configurable multi-agent reasoning workflows
- Routers: Query routing to optimal workflow configurations
- Runners: Evaluation orchestration and management
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies and missing dependencies
# Users should import specific modules as needed:
# from workflow_compiler.core import logger
# from workflow_compiler.benchmarks import MATHBenchmark
# etc.

__all__ = [
    "benchmarks",
    "compiler",
    "core",
    "dsl",
    "experiments",
    "routers",
    "runtime",
    "workflows",
]
