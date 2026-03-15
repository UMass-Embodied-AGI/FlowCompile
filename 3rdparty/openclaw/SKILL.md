---
name: openclaw-flowcompile
description: Use when an OpenClaw agent needs to stage an OpenClaw Lobster workflow, capture one interactive demo run, author a FlowCompile openclaw_lobster YAML config including judge prompts and workflow loops, validate it, and run get-latency, profile, and predict.
---

# OpenClaw FlowCompile

Use this skill when OpenClaw is running on its own machine and needs to prepare a workflow for FlowCompile.

## Rules

- Do not enter the GPU node.
- Do not manage model serving.
- Assume FlowCompile can reach models through the `model_config` in the authored flat YAML, whether that payload is written inline or via a referenced YAML file. That model config defines `endpoints.local_base_url` for `get-latency` and `endpoints.profile_base_url` for `profile`.
- Do not modify the source OpenClaw tree in place. Always work from the staged copy under `results/<experiment_id>/openclaw/`.

## Workflow

1. Stage the workflow:
   `flowcompile openclaw stage --workflow-file <path> --experiment-id <id>`
2. Run the interactive demo capture:
   `flowcompile openclaw demo-run --manifest results/<id>/openclaw/manifest.json [--args-json ...] [--env-json ...]`
3. If the workflow pauses for approval, surface the pause to the human and continue with:
   `flowcompile openclaw demo-resume --session results/<id>/openclaw/session/session.json [--approve yes|no] [--env-json ...]`
4. After the demo finishes, inspect the authoring bundle:
   `flowcompile openclaw analyze-demo --manifest results/<id>/openclaw/manifest.json`
5. Read [references/config-authoring.md](references/config-authoring.md), then author:
   `results/<id>/openclaw/flowcompile_openclaw.yaml`
6. Validate the config:
   `flowcompile openclaw validate-config --config results/<id>/openclaw/flowcompile_openclaw.yaml`
7. Run the pipeline:
   `flowcompile --config results/<id>/openclaw/flowcompile_openclaw.yaml get-latency`
   `flowcompile --config results/<id>/openclaw/flowcompile_openclaw.yaml profile`
   `flowcompile --config results/<id>/openclaw/flowcompile_openclaw.yaml predict`

## What To Read

- Read `results/<id>/openclaw/demo_analysis.json` after `analyze-demo`.
- Read [references/config-authoring.md](references/config-authoring.md) before writing the YAML.
- Use [configs/examples/flowcompile_openclaw_outlook.yaml](/proj/inf-scaling/efficient_foundation_models/workflow_compiler/codebase/configs/examples/flowcompile_openclaw_outlook.yaml) as the main example for inline `model_config`, `workflow_loops`, optional `predict_subagent_score_thresholds`, and `openclaw_agent_policies`.

## Authoring Notes

- Write judge prompts only after reviewing the captured demo samples.
- Infer `workflow_loops` from the demo analysis, then confirm loop counts with the human before finalizing them.
- Omit `predict_subagent_score_thresholds` unless the human explicitly wants pruning.
