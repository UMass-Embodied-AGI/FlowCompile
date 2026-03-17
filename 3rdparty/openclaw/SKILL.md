---
name: openclaw-flowcompile
description: Use when an OpenClaw agent needs to capture one interactive demo run from a bundled OpenClaw Lobster workflow, author a FlowCompile openclaw_lobster YAML config including judge prompts and workflow loops, validate it, and run get-latency, profile, and predict.
---

# OpenClaw FlowCompile

Use this skill when OpenClaw is running on its own machine and needs to optimize an existing bundled workflow with FlowCompile.

## Rules

- Do not enter the GPU node.
- Do not manage model serving.
- Always use `unlimited` budget during demo capture, including `demo-run` and any follow-up `demo-resume` steps.
- Assume FlowCompile can reach models through the `model_config` in the authored flat YAML, whether that payload is written inline or via a referenced YAML file. That model config defines `endpoints.local_base_url` for `get-latency` and `endpoints.profile_base_url` for `profile`.
- If any authored `openclaw_agent_policies[*].judge.mode` is `semantic_llm`, include `gpt-oss-120b` under `model_config.models`. FlowCompile profiling uses that alias as the default semantic judge model.
- Assume every profiled OpenClaw LLM step exposes a statically discoverable output schema under the workflow bundle's `prompts/` directory. FlowCompile validates payloads against those workflow-owned schemas before judging.
- Assume the workflow already exists as a runnable bundle directory under `workflows/<bundle>/`.
- Treat `workflows/<bundle>/flowcompile/` as the experiment root for all FlowCompile artifacts.

## Workflow

1. Run the interactive demo capture:
   `flowcompile openclaw demo-run --workflow-dir <bundle-dir> [--args-json ...] [--env-json ...]`
   - Prefer a workflow-provided debug mode when collecting the demo.
   - Always pass `budget_preset: "unlimited"` or the workflow's equivalent unlimited-budget setting in the capture inputs.
2. If the workflow pauses for approval, surface the pause to the human and continue with:
   `flowcompile openclaw demo-resume --workflow-dir <bundle-dir> [--approve yes|no] [--env-json ...]`
   - Keep the same unlimited-budget capture session and preserve the debug-mode path when the workflow supports it.
3. After the demo finishes, inspect the authoring bundle:
   `flowcompile openclaw analyze-demo --workflow-dir <bundle-dir>`
4. Read [references/config-authoring.md](references/config-authoring.md), then author:
   `<bundle-dir>/flowcompile/flowcompile_openclaw.yaml`
5. Validate the config:
   `flowcompile openclaw validate-config --workflow-dir <bundle-dir>`
6. Run the pipeline:
   `flowcompile --config <bundle-dir>/flowcompile/flowcompile_openclaw.yaml get-latency`
   `flowcompile --config <bundle-dir>/flowcompile/flowcompile_openclaw.yaml profile`
   `flowcompile --config <bundle-dir>/flowcompile/flowcompile_openclaw.yaml predict`

## What To Read

- Read `<bundle-dir>/flowcompile/demo_analysis.json` after `analyze-demo`.
- Read [references/config-authoring.md](references/config-authoring.md) before writing the YAML.
- Use [configs/examples/flowcompile_openclaw_outlook.yaml](configs/examples/flowcompile_openclaw_outlook.yaml) as the main example for inline `model_config`, `workflow_loops`, optional `predict_subagent_score_thresholds`, and `openclaw_agent_policies`.

## Authoring Notes

- `analyze-demo` depends on a completed demo export. A paused or partially completed session is not enough; finish the demo run before authoring YAML.
- For demo capture, first prefer a workflow-provided debug or dry-run path and keep the capture on `unlimited` budget.
- If no debug mode exists, tell the human that the demo will run in normal mode and ask them to make sure the demo exercises every workflow LLM step and relevant LLM-calling branch needed for FlowCompile capture.
- If the non-debug demo workflow has external side effects, surface that risk to the human before finishing the demo.
- Write judge prompts only after reviewing the captured demo samples.
- If `analyze-demo` reports any workflow LLM step with zero captured samples, reject that demo and capture another one before authoring YAML.
- If FlowCompile cannot statically resolve a profiled step's output schema from the workflow bundle, treat that as an authoring error and fix the workflow bundle before proceeding.
- Infer `workflow_loops` from the workflow structure exposed in the demo analysis, then confirm loop counts with the human before finalizing them.
- Treat observed demo counts in `candidate_workflow_loops` as hints only, not as proof that a loop does or does not exist.
- Omit `predict_subagent_score_thresholds` unless the human explicitly wants pruning.
- Set `profile_min_samples_per_agent: 20` in authored OpenClaw configs unless the human explicitly wants a different profiling floor.
- Before finalizing YAML, verify this checklist:
  - `required_fields` came from `demo_analysis.json`
  - every `required_fields` entry exists in that step's workflow-owned output schema
  - every `openclaw_agent_policies` key exactly matches a workflow step id
  - `profile_min_samples_per_agent` is set to `20`
  - confirmed `workflow_loops` counts with the human
  - included `gpt-oss-120b` in `model_config.models` whenever any semantic OpenClaw judge is present
