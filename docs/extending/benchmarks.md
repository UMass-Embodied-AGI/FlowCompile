# Adding a Benchmark

This guide documents the drop-in benchmark extension path used by FlowCompile. It is adapted from the maintainer notes in `workflow_compiler/benchmarks/ADDING_BENCHMARK.md`.

## Goal

Add a benchmark that works with:

- `flowcompile test`
- `flowcompile runtime infer`

For most cases, you only need one new module plus the dataset files it points to.

## 1. Create a Benchmark Module

Add:

```text
workflow_compiler/benchmarks/<name>.py
```

Use the repository template referenced by the maintainer guide as the starting point when available.

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

Put the benchmark data under `data/` and point `DEFAULT_SPLIT_PATHS` at the validate and test files.

## 4. Verify Registration

The benchmark registry is decorator-driven, so no central manual list should be necessary. Verify registration with:

```bash
python - <<'PY'
from workflow_compiler.benchmarks import list_benchmarks
for row in list_benchmarks(detailed=True):
    print(row["name"], row.get("aliases", []), row.get("workflow_type"), row.get("metric_name"))
PY
```

## 5. Validate End-to-End Usage

Example commands:

```bash
python -m workflow_compiler.core.cli test --dataset <alias-or-name>
python -m workflow_compiler.core.cli runtime infer --compiled <compiled_json> --workflow-type <workflow_type> --query "..."
```

Any alias listed in `ALIASES` should resolve to the same benchmark class.

## Related API

The benchmark registration helpers are documented in the API reference for `workflow_compiler.benchmarks.registry`.

