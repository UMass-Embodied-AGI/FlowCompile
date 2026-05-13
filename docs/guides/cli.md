# CLI Guide

The `flowcompile` command is the main user-facing entrypoint for the project.
This page is written manually instead of using autodoc because the CLI module
imports much more of the runtime stack than the docs build should require.

## Global Pattern

Most commands follow this shape:

```bash
flowcompile --config <path-to-config.yaml> <command> [subcommand] [options]
```

The config file is the source of truth for experiment identity, workflow type,
model config, split files, search axes, and search budgets. The current CLI
expects the flat schema `flowcompile.flat.v1`.

Global options:

- `--config`: path to a flat FlowCompile experiment YAML.
- `--verbose`: show detailed sub-step logs and repeated warnings.
- `--quiet`: keep warnings, errors, and final summaries only.
- `--plain`: disable interactive formatting and live progress updates.
- `--no-banner`: suppress the ASCII banner.

## Output Modes

The CLI uses a shared terminal presenter across commands. In an interactive
terminal it shows a FlowCompile ASCII banner once, concise step updates, and
progress bars for long-running phases.

Example:

```bash
flowcompile --verbose --config "$CONFIG" predict
flowcompile --plain --config "$CONFIG" run-all
```

## Core Pipeline Commands

### `get-latency`

Measure model latency for the configured `latency_models`. With an experiment
config, output defaults to
`results/<experiment_id>/01_profile/latency_benchmark.json`.

```bash
flowcompile --config "$CONFIG" get-latency
```

Useful overrides:

- `--models`
- `--output-json`
- `--prompt-file`
- `--batch-size` or `--batch-sizes`
- `--max-new-tokens`
- `--model-config-path`
- `--backend auto|vllm|openai`

### `prepare-data`

Run the reference DSL workflow and build the induced sub-agent dataset used by
profiling. This wraps `ground-truth` and `agent-dataset`.

```bash
flowcompile --config "$CONFIG" prepare-data
```

Useful overrides:

- `--task`
- `--llm`
- `--experiment-id`
- `--file-path`
- `--debug`
- `--model`
- `--max-samples`
- `--num-workers`

### `profile`

Profile induced sub-agent examples across the configured model and reasoning
budget choices.

```bash
flowcompile --config "$CONFIG" profile
```

Useful overrides:

- `--models`
- `--search-budgets`
- `--max-samples`
- `--max-concurrent`
- `--min-samples-per-agent`
- `--debug`

### `predict`

Compile candidate configurations with the structure-aware proxy and compute the
Pareto frontier. With an experiment config, the compiled payload defaults to
`results/<experiment_id>/02_compile/compiled_configs.json` and uses schema
`flowcompile.compiled.v2`.

```bash
flowcompile --config "$CONFIG" predict
```

Useful overrides:

- `--workflow-type`
- `--detailed-results`
- `--trace-data`
- `--latency-file`
- `--output-file`
- `--plot-file`
- `--include-all`
- `--prune-subagents` or `--no-prune-subagents`
- `--search-axes model budget structure`
- `--search-models`
- `--search-budgets`
- `--search-structures`
- `--search-agent-models agent=model1,model2`
- `--search-agent-budgets agent=100,1000,unlimited`

### `test`

Evaluate compiled Pareto configurations on the configured held-out split.

```bash
flowcompile --config "$CONFIG" test
```

Useful overrides:

- `--config-file`
- `--dataset`
- `--split`
- `--data-path`
- `--output-dir`
- `--pareto-sample-n`
- `--parallel`
- `--random-seed`
- `--start-idx`, `--end-idx`, and `--max-tasks`

`--pareto-sample-n -1` disables Pareto sampling and evaluates every compiled
Pareto config.

### `run-all`

Run the full pipeline in sequence.

```bash
flowcompile --config "$CONFIG" run-all
```

The `run-all` path shows a top-level stage progress tracker and keeps each stage summary brief.

## Runtime Commands

### `runtime infer`

Run a single query or JSONL batch against compiled configurations. Runtime
routing settings must be passed on the command line; the CLI rejects deprecated
YAML keys such as `runtime_strategy`, `runtime_budget`, and `runtime.alpha`.

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy preference \
  --budget 0.5
```

Useful runtime selector options:

- `--strategy preference` selects by weighted utility.
- `--strategy constraint` selects using explicit accuracy or latency constraints.
- `--strategy knn-router` builds a lightweight KNN router from profiling
  artifacts and selects from its candidate pool.
- `--budget` accepts either a numeric preference value or named presets:
  `low`, `medium`, `high`, and `xhigh`.
- `--min-accuracy` and `--max-latency` are only valid with
  `--strategy constraint`.
- `--queries` reads JSONL input for batch inference.
- `--compiled` can point at a non-canonical compiled config file.

## Experiments

### `experiments correlation`

Run correlation analysis between proxy-estimated and measured workflow
performance. With no extra arguments, the command derives inputs from the flat
config and expects test results under `results/<experiment_id>/03_test`.

```bash
flowcompile --config "$CONFIG" experiments correlation
```

Extra arguments are passed through to the underlying correlation script when
you need non-canonical paths.

## Working Style

- Keep one config file per experiment or benchmark variant.
- Start from a repository example in `configs/examples/`.
- Treat `run-all` as the convenience path and the individual commands as the
  debugging path.
- Use `--plain` for logs that need to be parsed by scripts or CI.
