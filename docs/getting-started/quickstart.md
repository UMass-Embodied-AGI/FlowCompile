# Quickstart

This is the standard FlowCompile workflow using one of the paper benchmark
configs. It runs the implementation version of the compiler pipeline: latency
benchmarking, trace and agent-data preparation, sub-agent profiling, Pareto
prediction, and held-out testing.

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

The stages write canonical artifacts under `results/<experiment_id>/`:

- `01_profile/latency_benchmark.json`
- `01_profile/aggregated_training_data.json`
- `01_profile/benchmark_*/detailed_results.json`
- `02_compile/compiled_configs.json`
- `02_compile/figures/compiled_latency_vs_score.png`
- `03_test/workflow_results_final.json`

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

Named preference budgets are also supported. They map to fixed values in the
implementation: `low` = `0.01`, `medium` = `0.5`, `high` = `0.9`, and
`xhigh` = `0.999`.

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy preference \
  --budget high
```

Constraint-based selection picks from the same compiled set:

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy constraint \
  --max-latency 20
```

Batch runtime inference reads JSONL queries:

```bash
flowcompile --config "$CONFIG" runtime infer \
  --queries data/math500_test.jsonl \
  --strategy preference \
  --budget medium
```

KNN-router selection is also available after profiling artifacts exist:

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy knn-router \
  --budget medium \
  --knn-k 20
```

## Analysis

To compare predicted and actual workflow accuracy and latency:

```bash
flowcompile --config "$CONFIG" experiments correlation
```

## What to Expect

- `get-latency` benchmarks the configured model set.
- `prepare-data` runs the reference workflow, scores traces, and builds the
  induced sub-agent dataset.
- `profile` measures sub-agent behavior over the configured model and budget
  choices.
- `predict` applies the structure-aware proxy and compiles the Pareto frontier.
- `test` validates compiled configurations on the held-out split. Use
  `test_pareto_sample_n: -1` to evaluate every Pareto config.
- `runtime infer` selects and runs a compiled configuration against a live
  query.
