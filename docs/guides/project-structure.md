# Project Structure

FlowCompile is organized as a Python package plus experiment configs, prepared
data, setup scripts, tests, and Sphinx docs.

## Top-Level Layout

```text
configs/              Local model config template and paper experiment configs
data/                 Prepared validate/test splits and prompt assets
docs/                 Sphinx documentation site
scripts/              Dataset and local serving helpers
tests/                Unit and smoke tests
workflow_compiler/    Main Python package
```

## Package Layout

```text
workflow_compiler/
  benchmarks/   Dataset-specific evaluation and registration
  compiler/     Latency, trace prep, agent data, profiling, prediction, validation
  core/         CLI, logging, data paths, LLM client helpers, analysis
  dsl/          Python DSL capture, auto-backward proxy, structures, runtime glue
  experiments/  Analysis commands such as correlation
  routers/      KNN routing implementations and helper conversions
  runtime/      Runtime config selection and inference helpers
  workflows/    Workflow definitions and extension points
```

## Where to Look First

- CLI entrypoint and flat config validation: `workflow_compiler/core/cli.py`
- Compiler pipeline and compiled JSON output: `workflow_compiler/compiler/`
- Structure-aware proxy and DSL capture: `workflow_compiler/dsl/`
- Built-in workflows: `workflow_compiler/workflows/`
- Benchmarks and metrics: `workflow_compiler/benchmarks/`
- Runtime selection and execution: `workflow_compiler/runtime/`
- Optional per-query routing: `workflow_compiler/routers/`

## Configuration and Results

- `configs/config.example.yaml` is the local model config template. Copy it to
  ignored `configs/config.yaml` before adding real keys or endpoints.
- `configs/examples/` contains benchmark-ready flat configs for GSM8K,
  MATH-500, HotpotQA, and LiveCodeBench.
- `results/<experiment_id>/01_profile/` stores latency benchmarks, reference
  workflow traces, induced agent data, and profiling results.
- `results/<experiment_id>/02_compile/` stores compiled configs and proxy plots.
- `results/<experiment_id>/03_test/` stores held-out validation results.
- `results/<experiment_id>/04_experiments/` stores analysis outputs such as
  proxy correlation metrics.
- `results/<experiment_id>/runtime/outputs/` stores runtime inference traces by
  default.

Generated `results/`, `runtime_outputs/`, `logs/`, docs build outputs, Python
caches, and local config files are ignored by git.

## Documentation Strategy

The docs site documents the project in three layers:

- Getting started pages for install, config, and quickstart.
- Guides that map the paper concepts onto the current CLI, workflows, and
  repository layout.
- Extension guides for adding benchmarks and Python DSL workflows.
- A curated API reference for stable or lightweight modules that do not require
  the full ML runtime stack to build.
