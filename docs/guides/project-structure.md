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
src/flowcompile/    Main Python package
```

## Package Layout

```text
src/flowcompile/
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

- CLI entrypoint and flat config validation: `src/flowcompile/core/cli.py`
- Compiler pipeline and compiled JSON output: `src/flowcompile/compiler/`
- Structure-aware proxy and DSL capture: `src/flowcompile/dsl/`
- Built-in workflows: `src/flowcompile/workflows/`
- Benchmarks and metrics: `src/flowcompile/benchmarks/`
- Runtime selection and execution: `src/flowcompile/runtime/`
- Optional per-query routing: `src/flowcompile/routers/`

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
