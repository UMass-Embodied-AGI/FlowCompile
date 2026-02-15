# Adding a Benchmark (Drop-In)

To add a new benchmark that works with:
- `flowcompile test`
- `flowcompile runtime infer`

you only need one new file.

## 1. Create a benchmark module

Create:
`workflow_compiler/benchmarks/<name>.py`

Use `templates/new_benchmark_template.py` as the starting point.

## 2. Implement the benchmark class

Requirements:
- Subclass `BaseBenchmark`
- Decorate the class with `@register_benchmark()`
- Define metadata attributes:
  - `BENCHMARK_NAME` (canonical ID)
  - `ALIASES` (accepted names)
  - `WORKFLOW_TYPE` (`math`, `gsm8k`, `hotpotqa`, `livecodebench`, or a new workflow type)
  - `METRIC_NAME` (`accuracy`, `f1`, `pass_at_1`, etc.)
  - `DEFAULT_SPLIT_PATHS` (e.g. `{"validate": "...", "test": "..."}`)

Required methods:
- `evaluate_problem(...)`
- `calculate_score(...)`
- `get_result_columns(...)`

Optional hooks (only when default behavior is not enough):
- `score_from_result(result)`
- `result_key(result)`
- `trace_key(trace)`

## 3. Add dataset files

Add your dataset files under `data/` and point `DEFAULT_SPLIT_PATHS` at them.

## 4. Verify registration

Run:
```bash
python - <<'PY'
from workflow_compiler.benchmarks import list_benchmarks
for row in list_benchmarks(detailed=True):
    print(row["name"], row.get("aliases", []), row.get("workflow_type"), row.get("metric_name"))
PY
```

No central registry edits are required.

## 5. Validate usage

Examples:
```bash
python -m workflow_compiler.core.cli test --dataset <alias-or-name>
python -m workflow_compiler.core.cli runtime infer --compiled <compiled_json> --workflow-type <workflow_type> --query "..."
```

Any alias in `ALIASES` resolves to the same benchmark class.
