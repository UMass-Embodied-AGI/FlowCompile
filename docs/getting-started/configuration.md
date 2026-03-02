# Configuration

FlowCompile expects a model configuration file and any required API keys or local endpoint settings before running the CLI.

## Create a Local Config

Start from the provided example:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Use `configs/config.yaml` for your local environment-specific values and keep the example file as the shareable template.

## Model and Endpoint Setup

You can point the project at hosted model APIs or local OpenAI-compatible endpoints. The exact values depend on the backend you run.

When reproducing the paper-style local setup, the repository README describes a `vllm` plus LiteLLM flow:

1. Start worker `vllm` servers.
2. Start judge `vllm` servers.
3. Update `scripts/setup_vllm/litellm_config_1worker1judge.yaml` with the correct worker and judge hosts.
4. Start the LiteLLM proxy.
5. Point entries in `configs/config.yaml` at the proxy endpoint and key.

## Configuration Inputs Used by the CLI

The unified CLI takes a benchmark configuration file via `--config`. The example benchmark configs live under `configs/examples/`:

- `configs/examples/flowcompile_hotpotqa.yaml`
- `configs/examples/flowcompile_gsm8k.yaml`
- `configs/examples/flowcompile_math500.yaml`
- `configs/examples/flowcompile_livecodebench.yaml`

Each experiment config references the model config file, dataset split files, search axes, and search budgets required by the flat config schema enforced in `workflow_compiler.core.cli`.

## Practical Notes

- Keep `configs/config.yaml` local if it contains secrets or machine-specific endpoints.
- Prefer copying and editing one of the example experiment YAMLs rather than starting from an empty file.
- If you use local inference servers, confirm the configured `base_url` and `api_key` match the proxy you actually started.

