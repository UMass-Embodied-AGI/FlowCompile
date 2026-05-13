# FlowCompile Documentation

FlowCompile is an optimizing compiler for structured LLM workflows. Given a
workflow graph, validation/profile data, and a design space over model choices,
reasoning budgets, and workflow structures, it compiles a reusable set of
workflow-level configurations that span the accuracy-latency Pareto frontier.

The implementation follows the paper pipeline closely:

1. Collect workflow traces with a high-capacity reference model.
2. Induce and filter sub-agent training examples from those traces.
3. Profile each sub-agent across the configured model and reasoning-budget space.
4. Compose sub-agent profiles through a structure-aware proxy.
5. Emit `flowcompile.compiled.v2` runtime configurations for validation,
   deployment-time selection, or optional routing.

## Core Capabilities

- Compile Pareto-optimal workflow configurations before deployment.
- Search model assignment, reasoning budget, and inferred workflow-structure
  choices with one flat experiment config.
- Define workflows in a Python DSL with reusable `AgentNode` and `ToolNode`
  components.
- Run latency benchmarking, data preparation, profiling, prediction, testing,
  runtime inference, and correlation analysis from the `flowcompile` CLI.
- Reuse compiled configurations with preference-based, constraint-based, or
  KNN-router selection at runtime.

## Start Here

- New to the project: begin with installation, configuration, and quickstart.
- Reproducing the paper workflow: read the compiler pipeline and benchmark
  workflow guides.
- Integrating experiments: use the CLI guide and project structure overview.
- Extending the codebase: read the benchmark and workflow extension guides.
- Looking for internals: use the curated API reference for stable modules.

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/configuration
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/compiler-pipeline
guides/benchmark-workflows
guides/cli
guides/project-structure

extending/benchmarks
extending/workflows
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

## Build the Docs Locally

Install the docs dependencies and run Sphinx from the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

The generated site will be written to `docs/_build/html`.
