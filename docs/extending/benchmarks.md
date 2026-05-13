# Adding a Benchmark

This guide documents the drop-in benchmark extension path used by FlowCompile.
Benchmarks own dataset loading, scoring, and result formatting; workflows own
execution structure.

## Goal

Add a benchmark that works with:

- `flowcompile test`
- `flowcompile runtime infer`
- benchmark discovery through `workflow_compiler.benchmarks.registry`

For most cases, you only need one new module plus the dataset files it points to.

## 1. Create a Benchmark Module

Add:

```text
workflow_compiler/benchmarks/<name>.py
```

Use the repository template referenced by the maintainer guide as the starting
point when available.

## 2. Implement the Benchmark Class

Your class must:

- subclass `BaseBenchmark`
- use `@register_benchmark()`
- define metadata fields:
  - `BENCHMARK_NAME`
  - `ALIASES`
  - `WORKFLOW_TYPE`
  - `METRIC_NAME`
  - `DEFAULT_SPLIT_PATHS`

Required methods:

- `evaluate_problem(...)`
- `calculate_score(...)`
- `get_result_columns(...)`

Optional hooks:

- `score_from_result(result)`
- `result_key(result)`
- `trace_key(trace)`

## 3. Add Dataset Files

Put the benchmark data under `data/` and point `DEFAULT_SPLIT_PATHS` at the
validate and test files:

```python
DEFAULT_SPLIT_PATHS = {
    "validate": "data/mybench_validate.jsonl",
    "test": "data/mybench_test.jsonl",
}
```

The flat experiment config should reference those same split files through
`validate_file` and `test_file`.

## 4. Verify Registration

The benchmark registry is decorator-driven, so no central manual list should be
necessary. Verify registration with:

```bash
python - <<'PY'
from workflow_compiler.benchmarks import list_benchmarks
for row in list_benchmarks(detailed=True):
    print(row["name"], row.get("aliases", []), row.get("workflow_type"), row.get("metric_name"))
PY
```

## 5. Validate End-to-End Usage

Add or copy a flat config under `configs/examples/`:

```yaml
schema_version: "flowcompile.flat.v1"
experiment_id: "mybench"
workflow_type: "math"
dataset: "MyBench"
model_config: "configs/config.yaml"
validate_file: "data/mybench_validate.jsonl"
test_file: "data/mybench_test.jsonl"
search_axes: ["model", "budget", "structure"]
search_budgets: [100, 500, 1000]
```

Use one of the currently supported CLI workflow types: `math`, `gsm8k`,
`hotpotqa`, or `livecodebench`. A genuinely new workflow type also needs the
workflow extension steps, plus CLI/runtime support for that type.

Then validate the usual pipeline:

```bash
CONFIG=configs/examples/flowcompile_mybench.yaml

flowcompile --config "$CONFIG" get-latency
flowcompile --config "$CONFIG" prepare-data
flowcompile --config "$CONFIG" profile
flowcompile --config "$CONFIG" predict
flowcompile --config "$CONFIG" test
flowcompile --config "$CONFIG" runtime infer \
  --query "..." \
  --strategy preference \
  --budget medium
```

Any alias listed in `ALIASES` should resolve to the same benchmark class.

## Related API

The benchmark registration helpers are documented in the API reference for `workflow_compiler.benchmarks.registry`.
