# Project Structure

FlowCompile is organized as a Python package with a small set of top-level directories for configuration, data, scripts, tests, and documentation.

## Top-Level Layout

```text
configs/              Example and local configuration files
data/                 Datasets and supporting assets
docs/                 Sphinx documentation site
scripts/              Setup helpers and utility scripts
tests/                Test suite
workflow_compiler/    Main Python package
```

## Package Layout

```text
workflow_compiler/
  benchmarks/   Dataset-specific evaluation and registration
  compiler/     Data prep, profiling, latency, prediction, validation
  core/         CLI, logging, data paths, LLM client helpers, analysis
  dsl/          Python DSL models, execution, structure inference, runtime glue
  experiments/  Analysis commands such as correlation
  routers/      Routing implementations and helpers
  runtime/      Runtime config selection and inference helpers
  workflows/    Workflow definitions and extension points
```

## Where to Look First

- Extending datasets: `workflow_compiler/benchmarks/`
- Extending workflows: `workflow_compiler/workflows/`
- Understanding runtime selection: `workflow_compiler/runtime/`
- Understanding the DSL surface: `workflow_compiler/dsl/`
- Understanding the command-line entrypoint: `workflow_compiler/core/cli.py`

## Configuration and Results

- `configs/config.example.yaml` is the local model config template.
- `configs/examples/` contains benchmark-ready CLI configs.
- `results/` and `runtime_outputs/` are used for generated artifacts during experiments and runtime execution.

## Documentation Strategy

The docs site documents the project in three layers:

- Getting started pages for install, config, and quickstart.
- Guides for the CLI and repository layout.
- A curated API reference for stable or lightweight modules that do not require the full ML runtime stack to build.

