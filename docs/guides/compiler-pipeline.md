# Compiler Pipeline

This guide maps the FlowCompile paper pipeline to the current implementation.
The short version: FlowCompile compiles structured LLM workflows before
deployment by profiling sub-agents once, composing those measurements with a
structure-aware proxy, and writing a reusable Pareto set of runtime
configurations.

## Inputs

The compiler needs:

- A Python DSL workflow from `src/flowcompile/workflows/`.
- A labeled validation/profile split.
- A local model config from `configs/config.yaml`.
- A flat experiment config with `search_axes` and `search_budgets`.
- Latency measurements for the model set.

The built-in workflow types are `math`, `gsm8k`, `hotpotqa`, and
`livecodebench`.

## Stage 0: Latency Benchmark

```bash
flowcompile --config "$CONFIG" get-latency
```

This writes `results/<experiment_id>/01_profile/latency_benchmark.json`.
The default config path is canonical because later stages resolve latency from
that location. The benchmark can use local vLLM serving or an OpenAI-compatible
backend, depending on `latency_backend` and the model config.

## Stage 1: Trace Collection and Agent Data

```bash
flowcompile --config "$CONFIG" prepare-data
```

`prepare-data` runs two stages:

- `ground-truth`: executes the full DSL workflow on the validation split using
  a high-capacity reference model such as `gpt-5-mini`.
- `agent-dataset`: applies an LLM-as-a-judge filter to workflow traces and
  produces induced sub-agent examples.

The main output used by later stages is
`results/<experiment_id>/01_profile/aggregated_training_data.json`.

## Stage 2: Sub-Agent Profiling

```bash
flowcompile --config "$CONFIG" profile
```

Profiling evaluates each induced sub-agent dataset across candidate model and
reasoning-budget settings. The current implementation uses settings formatted
as `<model>_budget_<budget>`, with `budget` values coming from
`search_budgets`.

The detailed profiling output is written under
`results/<experiment_id>/01_profile/benchmark_*/detailed_results.json`.

## Stage 3: Compositional Prediction

```bash
flowcompile --config "$CONFIG" predict
```

Prediction is the compiler step. It:

1. Consolidates profiling results, trace metadata, and model-latency data.
2. Aggregates per-sub-agent accuracy, input tokens, output tokens, and latency.
3. Optionally Pareto-prunes locally dominated sub-agent settings.
4. Enumerates inferred workflow structures and active sub-agent settings.
5. Runs the DSL `backward` proxy to estimate workflow accuracy and latency.
6. Applies non-dominated sorting to keep Pareto configurations.

The output schema is `flowcompile.compiled.v2`:

```json
{
  "schema_version": "flowcompile.compiled.v2",
  "workflow_type": "math",
  "metadata": {
    "search_space": {
      "search_axes": ["budget", "model", "structure"]
    }
  },
  "configs": []
}
```

Each config contains a `config_id`, `structure_id`, estimated metrics, and
per-agent runtime settings.

## Proxy Semantics

The implementation uses the Python DSL's default auto-backward proxy unless a
workflow overrides `backward(payload)`.

Accuracy is composed from profiled sub-agent accuracies according to the
captured workflow graph:

- Direct dependencies compose sequentially.
- Candidate lists compose as disjunctive branches, so any correct branch can
  support a correct downstream answer.
- Tool nodes are treated as deterministic pass-through stages for accuracy.
- Captured loop-break patterns such as `if test_out["test_passed"]: break`
  support bounded retry estimation for LiveCodeBench-style repair.

Latency currently uses the sequential edge execution model. It sums active
sub-agent latencies, with retry latency adjusted by the expected number of
repair attempts. Other execution models would require a different latency
composer, but no re-profiling of sub-agents is required conceptually.

## Stage 4: Held-Out Testing

```bash
flowcompile --config "$CONFIG" test
```

Testing executes compiled Pareto configurations on the configured held-out
split. The config key `test_pareto_sample_n` controls how many Pareto configs
are evaluated; `-1` evaluates all Pareto configs.

The main output is `results/<experiment_id>/03_test/workflow_results_final.json`.

## Runtime Use

After compilation, deployment does not search the full design space again.
Runtime selection chooses from the compiled config set:

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy preference \
  --budget medium
```

Supported strategies:

- `preference`: maximize a weighted accuracy-latency utility.
- `constraint`: satisfy `--min-accuracy` and/or `--max-latency`.
- `knn-router`: build a query-neighbor router from profiling artifacts, then
  select among the router-produced candidate configs.

## Analysis

The paper validates the proxy with rank and calibration metrics. The
implementation exposes the correlation analysis through:

```bash
flowcompile --config "$CONFIG" experiments correlation
```

This compares proxy-estimated and measured workflow accuracy/latency using the
compiled and tested artifacts for the same experiment.
