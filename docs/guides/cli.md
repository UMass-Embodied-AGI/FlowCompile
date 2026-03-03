# CLI Guide

The `flowcompile` command is the main user-facing entrypoint for the project. This page is written manually instead of using autodoc because the CLI module imports much more of the runtime stack than the docs build should require.

## Global Pattern

Most commands follow this shape:

```bash
flowcompile --config <path-to-config.yaml> <command> [subcommand] [options]
```

The config file is the source of truth for experiment identity, workflow type, model config, split files, search axes, and search budgets.

## Output Modes

The CLI now uses a shared terminal presenter across commands. In an interactive terminal it shows a FlowCompile ASCII banner once, concise step updates, and progress bars for long-running phases.

Global output flags:

- `--verbose` shows detailed sub-step logs and repeated warnings.
- `--quiet` suppresses banner, progress bars, and routine status lines while keeping warnings, errors, and final summaries.
- `--plain` disables interactive formatting and live progress updates, emitting stable newline-based output instead.
- `--no-banner` suppresses the ASCII banner even in interactive TTY mode.

Example:

```bash
flowcompile --verbose --config "$CONFIG" predict
flowcompile --plain --config "$CONFIG" run-all
```

## Core Pipeline Commands

### `get-latency`

Measure model latency for the configured model set.

```bash
flowcompile --config "$CONFIG" get-latency
```

### `prepare-data`

Prepare ground-truth labels and agent dataset artifacts used by later pipeline stages.

```bash
flowcompile --config "$CONFIG" prepare-data
```

### `profile`

Profile workflow performance across the configured search budgets.

```bash
flowcompile --config "$CONFIG" profile
```

### `predict`

Compile candidate configurations and compute the Pareto frontier.

```bash
flowcompile --config "$CONFIG" predict
```

### `test`

Evaluate compiled configurations on the configured test split.

```bash
flowcompile --config "$CONFIG" test
```

### `run-all`

Run the full pipeline in sequence.

```bash
flowcompile --config "$CONFIG" run-all
```

The `run-all` path shows a top-level stage progress tracker and keeps each stage summary brief.

## Runtime Commands

### `runtime infer`

Run a single query against compiled configurations.

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy preference \
  --budget 0.5
```

Useful runtime selector options:

- `--strategy preference` selects by weighted utility.
- `--strategy constraint` selects using explicit accuracy or latency constraints.
- `--budget` accepts either a numeric preference value or named presets such as `low`, `medium`, `high`, and `xhigh`.

## Experiments

### `experiments correlation`

Run correlation analysis between predicted and actual performance.

```bash
flowcompile --config "$CONFIG" experiments correlation
```

## Working Style

- Keep one config file per experiment or benchmark variant.
- Start from a repository example in `configs/examples/`.
- Treat `run-all` as the convenience path and the individual commands as the debugging path.
