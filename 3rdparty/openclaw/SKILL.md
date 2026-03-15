---
name: openclaw-flowcompile
description: Use when an OpenClaw agent needs to capture one interactive demo run from a bundled OpenClaw Lobster workflow, author a FlowCompile openclaw_lobster YAML config including judge prompts and workflow loops, validate it, and run get-latency, profile, and predict.
---

# OpenClaw FlowCompile

Use this skill when OpenClaw is running on its own machine and needs to optimize an existing bundled workflow with FlowCompile.

## Rules

- Do not enter the GPU node.
- Do not manage model serving.
- Assume FlowCompile can reach models through the `model_config` in the authored flat YAML, whether that payload is written inline or via a referenced YAML file. That model config defines `endpoints.local_base_url` for `get-latency` and `endpoints.profile_base_url` for `profile`.
- Assume the workflow already exists as a runnable bundle directory under `workflows/<bundle>/`.
- Treat `workflows/<bundle>/flowcompile/` as the experiment root for all FlowCompile artifacts.

## Workflow

1. Run the interactive demo capture:
   `flowcompile openclaw demo-run --workflow-dir <bundle-dir> [--args-json ...] [--env-json ...]`
2. If the workflow pauses for approval, surface the pause to the human and continue with:
   `flowcompile openclaw demo-resume --workflow-dir <bundle-dir> [--approve yes|no] [--env-json ...]`
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

- Write judge prompts only after reviewing the captured demo samples.
- Infer `workflow_loops` from the demo analysis, then confirm loop counts with the human before finalizing them.
- Omit `predict_subagent_score_thresholds` unless the human explicitly wants pruning.
