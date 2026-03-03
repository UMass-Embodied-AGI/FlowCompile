# FlowCompile Documentation

FlowCompile is an agentic LLM workflow compiler that computes an accuracy-latency Pareto frontier at compile time and uses it to pick practical runtime configurations later.

This site is the project documentation home for setup, configuration, CLI usage, extension guides, and a curated API reference.

## Core Capabilities

- Compile Pareto-optimal workflow configurations before deployment.
- Define workflows in a Python DSL with explicit structure and reusable agents.
- Run the same project through a single `flowcompile` CLI for benchmarking, profiling, compilation, validation, and runtime inference.
- Extend the system with new benchmarks and workflows without rebuilding the whole documentation stack.

## Start Here

- New to the project: begin with the installation and quickstart guides.
- Integrating the CLI into experiments: use the CLI guide and project structure overview.
- Extending the codebase: read the benchmark and workflow extension guides.
- Looking for internals: use the curated API reference for the stable modules documented here.

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
