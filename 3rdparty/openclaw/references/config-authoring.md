# FlowCompile OpenClaw Config Authoring

Use this reference after `flowcompile openclaw analyze-demo --workflow-dir <bundle-dir>`.

## Required Keys

Every authored config must include:

- `schema_version: "flowcompile.flat.v1"`
- `experiment_id`
- `experiment_root`
- `workflow_type: "openclaw_lobster"`
- `model_config`
- `openclaw_lobster_workflow_file`
- `profile_training_data`
- `predict_trace_data`
- `search_axes`
- `search_models`
- `search_budgets`
- `profile_models`
- `latency_models`

`model_config` may be either:

- a string path to a standalone model-config YAML, or
- an inline mapping embedded directly in the flat config

The model-config payload must define shared OpenAI-compatible endpoints:

```yaml
endpoints:
  local_base_url: "http://127.0.0.1:4000"
  profile_base_url: "http://profile-host:4000"
models:
  qwen35-9b-awq:
    api_type: "openai"
    api_key: "sk-proxy-demo-key"
    hf_model_name: "QuantTrio/Qwen3.5-9B-AWQ"
  gpt-oss-120b:
    api_type: "openai"
    api_key: "sk-proxy-demo-key"
    hf_model_name: "openai/gpt-oss-120b"
```

Rules:

- `get-latency` uses `endpoints.local_base_url`.
- `profile` uses `endpoints.profile_base_url` for both sub-agent calls and judge calls.
- If any `openclaw_agent_policies[*].judge.mode` is `semantic_llm`, `model_config.models` must include `gpt-oss-120b`. FlowCompile profiling uses that alias as the default semantic judge model.
- For `workflow_type: "openclaw_lobster"`, every profiled LLM step must expose a statically discoverable output schema under the workflow bundle's `prompts/` directory. FlowCompile validates payloads against those workflow-owned schemas before judging.
- Keep model aliases and `hf_model_name` mappings under `models`.
- Do not repeat endpoint URLs under every model entry in new configs.
- FlowCompile resolves `model_config` to an in-memory mapping after loading the flat config, even if the author wrote it as a path.

Inline example:

```yaml
model_config:
  endpoints:
    local_base_url: "http://127.0.0.1:4000"
    profile_base_url: "http://profile-host:4000"
  models:
    qwen35-9b-awq:
      api_type: "openai"
      api_key: "sk-proxy-demo-key"
      hf_model_name: "QuantTrio/Qwen3.5-9B-AWQ"
    gpt-oss-120b:
      api_type: "openai"
      api_key: "sk-proxy-demo-key"
      hf_model_name: "openai/gpt-oss-120b"
```

Useful optional keys:

- `judge_policies`
- `openclaw_agent_policies`
- `workflow_loops`
- `predict_subagent_score_thresholds`
- `profile_min_samples_per_agent`
- `profile_max_concurrent`

Recommended default:

- Set `profile_min_samples_per_agent: 20` for OpenClaw profiling unless you are intentionally running a smaller smoke test.

## judge_policies

Use `judge_policies` for non-OpenClaw prompt-based judges such as math, HotpotQA, programmer, and `sc_ensemble_*`.

Shape:

```yaml
judge_policies:
  <agent_name>:
    mode: semantic_llm
    prompt: |
      full prompt text
```

Rules:

- `mode` is currently `semantic_llm`.
- Provide one entry per prompt-judged agent used by the workflow.
- Common placeholders:
  - `{ground_truth}`
  - `{model_output}`
- Additional placeholders by agent type:
  - `programmer`: `{exec_output}`
  - `answer_generate`: `{question}`
  - `sc_ensemble_math`: `{ground_truth_solution}`, `{predicted_solution}`, `{ground_truth_output}`, `{model_output}`
  - `sc_ensemble_hotpotqa`: same as above plus `{question}`
  - `sc_ensemble_livecodebench`: same as above plus `{problem}`

## Path Writing

- Prefer the bundle-local relative paths suggested in `<bundle-dir>/flowcompile/demo_analysis.json`.
- For bundled OpenClaw workflows, set `experiment_root: "."` because the authored config lives inside `<bundle-dir>/flowcompile/`.
- The config validator accepts paths that are either valid from the current working directory or valid relative to the config file location.
- Keep all FlowCompile artifacts under `<bundle-dir>/flowcompile/`; do not point back to a separate `results/<id>/...` tree.

## openclaw_agent_policies

Write one policy per profiled LLM step. The keys must exactly match workflow step ids.

Shape:

```yaml
openclaw_agent_policies:
  <agent_name>:
    required_fields: ["field_name"]
    judge:
      mode: strict_exact | semantic_llm
      prompt: |
        full prompt text
```

Rules:

- Derive `required_fields` from the captured JSON outputs in `demo_analysis.json`.
- Every `required_fields` entry must exist in the profiled step's resolved workflow schema; FlowCompile rejects unknown schema properties.
- Use `strict_exact` when the output is an exact categorical/string field and the right answer should match exactly after normalization.
- Use `semantic_llm` for summaries, free-form text, or multi-constraint outputs.
- `strict_exact` does not require an LLM judge call. Any `semantic_llm` OpenClaw judge requires `gpt-oss-120b` to be present in `model_config.models`.
- For `semantic_llm`, write the full `prompt` after reviewing the demo examples for that agent.
- The semantic prompt can reference:
  - `{input_prompt}`
  - `{required_fields}`
  - `{ground_truth_field}`
  - `{predicted_field}`
  - `{ground_truth_json}`
  - `{predicted_json}`

## OpenClaw Authoring Checklist

Before finalizing the YAML:

- Read `<bundle-dir>/flowcompile/demo_analysis.json`.
- Derive `required_fields` directly from the agent analysis.
- Confirm the workflow bundle exposes statically discoverable `prompts/*.schema.json` files for every profiled OpenClaw step.
- Set `profile_min_samples_per_agent: 20` unless the human explicitly wants a smaller or larger profiling floor.
- Confirm `workflow_loops` counts with the human instead of blindly copying observed counts.
- Include `gpt-oss-120b` in `model_config.models` whenever any OpenClaw semantic judge is present.

## workflow_loops

Use `workflow_loops` when repeated LLM calls materially change workflow latency.

Shape:

```yaml
experiment_root: "."
workflow_loops:
  - name: "email_loop"
    count: 20
    map_nodes: ["summarize_each", "classify"]
    reduce_node: "overview"
```

Rules:

- `count` multiplies latency for every node in `map_nodes`.
- `reduce_node`, if present, is counted once.
- `reduce_node` must be a `map_reduce` or `reduce` node.
- A node cannot appear in more than one loop.
- `candidate_workflow_loops` in `demo_analysis.json` are structural authoring candidates. Their observed counts are hints from the captured demo, not the rule for whether the loop exists.
- If `analyze-demo` rejects the bundle because any workflow LLM step has zero captured samples, collect another demo before authoring the YAML.
- Start from the candidate loops in `demo_analysis.json`, then confirm or edit loop counts with the human.

## predict_subagent_score_thresholds

This block is optional.

Shape:

```yaml
predict_subagent_score_thresholds:
  summarize_each: 0.4
  classify: 0.4
```

Rules:

- Keys must exactly match workflow subagent names.
- Values must be floats in `[0, 1]`.
- Thresholding removes low-scoring subagent settings before workflow composition.
- If a threshold removes every candidate for a subagent, `predict` fails.
- Omit this block unless the human explicitly wants pruning.

## Example

See [configs/examples/flowcompile_openclaw_outlook.yaml](/proj/inf-scaling/efficient_foundation_models/workflow_compiler/codebase/configs/examples/flowcompile_openclaw_outlook.yaml).
