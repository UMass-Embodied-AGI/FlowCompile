# Quickstart

This is the standard FlowCompile workflow using one of the example benchmark configs.

## Choose a Config

```bash
CONFIG=configs/examples/flowcompile_math500.yaml
```

## Run the Canonical Pipeline

```bash
flowcompile --config "$CONFIG" get-latency
flowcompile --config "$CONFIG" prepare-data
flowcompile --config "$CONFIG" profile
flowcompile --config "$CONFIG" predict
flowcompile --config "$CONFIG" test
```

The end-to-end shortcut runs the same stages in order:

```bash
flowcompile --config "$CONFIG" run-all
```

## Runtime Inference

Once you have compiled configurations, you can run a single query through the runtime selector:

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy preference \
  --budget 0.5
```

Named preference budgets are also supported:

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy preference \
  --budget high
```

## Analysis

To compare predicted and actual workflow accuracy and latency:

```bash
flowcompile --config "$CONFIG" experiments correlation
```

## What to Expect

- `get-latency` benchmarks the configured model set.
- `prepare-data` prepares ground-truth and agent profiling data.
- `profile` measures workflow behavior over the configured search budgets.
- `predict` compiles the Pareto frontier.
- `test` validates compiled configurations on the test split.
- `runtime infer` selects and runs a configuration against a live query.

