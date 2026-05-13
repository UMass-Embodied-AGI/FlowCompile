# Configuration

FlowCompile uses two YAML inputs:

- A local model config such as `configs/config.yaml`, which contains API keys,
  endpoints, deployment names, and model metadata.
- A flat experiment config such as
  `configs/examples/flowcompile_math500.yaml`, which describes the benchmark,
  split files, search space, and command defaults.

## Create a Local Config

Start from the provided example:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Use `configs/config.yaml` for your local environment-specific values and keep the example file as the shareable template.

## Model and Endpoint Setup

The model config has a top-level `models` map. Each entry is keyed by the alias
used in experiment configs and compiled runtime settings.

Hosted Azure-style entries use fields like:

```yaml
models:
  gpt-5-mini:
    api_type: "azure"
    azure_endpoint: "https://YOUR_AZURE_ENDPOINT"
    azure_deployment: "gpt-5-mini-2024-07-18"
    api_key: "YOUR_AZURE_API_KEY"
    api_version: "2024-10-21"
```

Local or proxy-hosted models can be exposed through an OpenAI-compatible
endpoint:

```yaml
models:
  qwen3-4b:
    api_type: "openai"
    base_url: "http://127.0.0.1:4000"
    api_key: "YOUR_PROXY_KEY"
    hf_model_name: "Qwen/Qwen3-4B"
    enable_thinking_budget: true
    thinking_budget_reasoning_parser: "qwen3"
```

`hf_model_name` is important when FlowCompile derives search model aliases from
the latency benchmark model list.

When reproducing the paper-style local setup, use a vLLM plus LiteLLM flow:

1. Start worker `vllm` servers.
2. Start judge `vllm` servers.
3. Update `scripts/setup_vllm/litellm_config_1worker1judge.yaml` with the correct worker and judge hosts.
4. Start the LiteLLM proxy.
5. Point entries in `configs/config.yaml` at the proxy endpoint and key.

## Flat Experiment Config

The unified CLI takes an experiment config through `--config`. The current
schema is `flowcompile.flat.v1`; nested legacy sections such as `compile`,
`runtime`, `models`, `defaults`, and `shared` are rejected by the CLI.

Required flat keys:

- `schema_version`: must be `flowcompile.flat.v1`
- `experiment_id`: result directory name under `results/`
- `workflow_type`: `math`, `gsm8k`, `hotpotqa`, or `livecodebench`
- `dataset`: benchmark name or alias
- `model_config`: path to the local model config
- `validate_file` and `test_file`: split files
- `search_axes`: any subset of `model`, `budget`, and `structure`
- `search_budgets`: discrete reasoning budgets for profiling and search

The paper benchmark configs live under `configs/examples/`:

- `configs/examples/flowcompile_hotpotqa.yaml`
- `configs/examples/flowcompile_gsm8k.yaml`
- `configs/examples/flowcompile_math500.yaml`
- `configs/examples/flowcompile_livecodebench.yaml`

Optional flat keys control command defaults. Common examples include:

- `ground_truth_llm` and `agent_dataset_model`
- `latency_models`, `latency_backend`, and `latency_batch_size`
- `min_samples_per_agent`, `profile_max_samples`, and
  `profile_max_concurrent`
- `predict_include_all_configs` and `predict_prune_subagents`
- `test_parallel`, `test_max_tasks`, `test_random_seed`, and
  `test_pareto_sample_n`
- `runtime_output_dir`
- `correlation_output_dir` and `correlation_optimize_calibration`

Runtime routing preferences are intentionally not read from YAML. Pass them on
the `runtime infer` command line with `--strategy`, `--budget`,
`--min-accuracy`, or `--max-latency`.

## Search Space

The paper searches three axes:

- `model`: model alias assignment per sub-agent.
- `budget`: reasoning-token budget per sub-agent call.
- `structure`: inferred workflow-structure variants from the Python DSL.

By default, search models are derived from `latency_models` by matching each
Hugging Face model name to exactly one `hf_model_name` entry in the model
config. You can narrow the search at prediction time with CLI flags:

```bash
flowcompile --config "$CONFIG" predict \
  --search-axes model budget structure \
  --search-models qwen3-4b qwen3-8b \
  --search-budgets 200 1000 4000 \
  --search-agent-models sc_ensemble=qwen3-8b,qwen3-14b
```

## Practical Notes

- Keep `configs/config.yaml` local if it contains secrets or machine-specific endpoints.
- Prefer copying and editing one of the example experiment YAMLs rather than
  starting from an empty file.
- If you use local inference servers, confirm the configured `base_url` and
  `api_key` match the proxy you actually started.
- LiveCodeBench configs also need `livecodebench_public_test_file` for public
  test execution metadata.
