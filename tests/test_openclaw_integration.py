from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow_compiler.integration.openclaw import (
    analyze_openclaw_demo,
    stage_openclaw_workspace,
    validate_openclaw_config_payload,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_workflow_root(root: Path) -> Path:
    workflow_path = root / "workflows" / "outlook.lobster.yaml"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(
        "\n".join(
            [
                "name: outlook-past-24h",
                "version: 1",
                "steps:",
                "  - id: fetch_full_bodies",
                "    command: ./bin/outlook_fetch_full_bodies",
                "  - id: summarize_each",
                "    stdin: $fetch_full_bodies.stdout",
                "    command: |",
                "      cd /home/junyan/.openclaw/workspace",
                "      ./bin/outlook_llm_summarize_each",
                "  - id: classify",
                "    stdin: $fetch_full_bodies.stdout",
                "    command: |",
                "      cat <<'__SUM__'",
                "      $summarize_each.stdout",
                "      __SUM__",
                "      ./bin/outlook_llm_classify",
                "  - id: overview",
                "    stdin: $fetch_full_bodies.stdout",
                "    command: |",
                "      cat <<'__SUM__'",
                "      $summarize_each.stdout",
                "      __SUM__",
                "      cat <<'__CLS__'",
                "      $classify.stdout",
                "      __CLS__",
                "      ./bin/outlook_llm_overview",
                "  - id: ask_questions",
                "    stdin: $fetch_full_bodies.stdout",
                "    command: |",
                "      cat <<'__OVERVIEW__'",
                "      $overview.stdout",
                "      __OVERVIEW__",
                "      ./bin/outlook_llm_questions",
                "  - id: draft_replies",
                "    stdin: $fetch_full_bodies.stdout",
                "    command: |",
                "      cat <<'__OVERVIEW__'",
                "      $overview.stdout",
                "      __OVERVIEW__",
                "      cat <<'__ASK__'",
                "      $ask_questions.stdout",
                "      __ASK__",
                "      ./bin/outlook_llm_draft_replies",
            ]
        ),
        encoding="utf-8",
    )
    script_path = root / "bin" / "run_outlook_past_24h"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "\n".join(
            [
                "#!/home/junyan/miniconda3/bin/python",
                'ROOT="/home/junyan/.openclaw/workspace"',
                'FLOWCOMPILE_ROOT="/home/junyan/code/FlowCompile"',
            ]
        ),
        encoding="utf-8",
    )
    return workflow_path


def _training_payload() -> dict:
    return {
        "training_data": [
            {
                "agent_name": "summarize_each",
                "raw_llm_prompt": "summarize prompt",
                "processed_output": '{"summary":"s1"}',
                "raw_llm_output": '{"summary":"s1"}',
            },
            {
                "agent_name": "summarize_each",
                "raw_llm_prompt": "summarize prompt",
                "processed_output": '{"summary":"s2"}',
                "raw_llm_output": '{"summary":"s2"}',
            },
            {
                "agent_name": "classify",
                "raw_llm_prompt": "classify prompt",
                "processed_output": '{"category":"reply"}',
                "raw_llm_output": '{"category":"reply"}',
            },
            {
                "agent_name": "classify",
                "raw_llm_prompt": "classify prompt",
                "processed_output": '{"category":"reply"}',
                "raw_llm_output": '{"category":"reply"}',
            },
            {
                "agent_name": "overview",
                "raw_llm_prompt": "overview prompt",
                "processed_output": '{"overview_paragraph":"overview"}',
                "raw_llm_output": '{"overview_paragraph":"overview"}',
            },
            {
                "agent_name": "ask_questions",
                "raw_llm_prompt": "question prompt",
                "processed_output": '{"question":"q1"}',
                "raw_llm_output": '{"question":"q1"}',
            },
            {
                "agent_name": "ask_questions",
                "raw_llm_prompt": "question prompt",
                "processed_output": '{"question":"q2"}',
                "raw_llm_output": '{"question":"q2"}',
            },
            {
                "agent_name": "draft_replies",
                "raw_llm_prompt": "draft prompt",
                "processed_output": '{"draft_body":"body1"}',
                "raw_llm_output": '{"draft_body":"body1"}',
            },
            {
                "agent_name": "draft_replies",
                "raw_llm_prompt": "draft prompt",
                "processed_output": '{"draft_body":"body2"}',
                "raw_llm_output": '{"draft_body":"body2"}',
            },
        ]
    }


def test_stage_openclaw_workspace_rewrites_stale_paths(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(repo_root)
    workflow_path = _write_workflow_root(tmp_path / "source_openclaw")

    manifest_path = stage_openclaw_workspace(str(workflow_path), "exp_stage")

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    staged_workflow = Path(manifest["staged_workflow_file"])
    staged_script = Path(manifest["staged_root"]) / "bin" / "run_outlook_past_24h"
    staged_workflow_text = staged_workflow.read_text(encoding="utf-8")
    staged_script_text = staged_script.read_text(encoding="utf-8")

    assert "FLOWCOMPILE_OPENCLAW_STEP_ID=summarize_each" in staged_workflow_text
    assert "/home/junyan/.openclaw/workspace" not in staged_workflow_text
    assert "#!/usr/bin/env python3" in staged_script_text
    assert "/home/junyan/code/FlowCompile" not in staged_script_text


def test_analyze_openclaw_demo_emits_candidate_loops(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(repo_root)
    workflow_path = _write_workflow_root(tmp_path / "source_openclaw")
    manifest_path = stage_openclaw_workspace(str(workflow_path), "exp_analyze")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _write_json(Path(manifest["training_data_path"]), _training_payload())

    analysis_path = analyze_openclaw_demo(str(manifest_path))
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))

    assert analysis["training_data_summary"]["counts_by_agent"]["summarize_each"] == 2
    loops = analysis["candidate_workflow_loops"]
    assert any(loop["reduce_node"] == "overview" for loop in loops if "reduce_node" in loop)
    assert any(loop["map_nodes"] == ["ask_questions", "draft_replies"] for loop in loops)
    assert analysis["agents"]["overview"]["required_fields_intersection"] == ["overview_paragraph"]


def test_validate_openclaw_config_payload_accepts_valid_config(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(repo_root)
    workflow_path = _write_workflow_root(tmp_path / "source_openclaw")
    manifest_path = stage_openclaw_workspace(str(workflow_path), "exp_validate")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _write_json(Path(manifest["training_data_path"]), _training_payload())
    model_config = repo_root / "configs" / "config.qwen35.local.yaml"
    model_config.parent.mkdir(parents=True, exist_ok=True)
    model_config.write_text("models: {}\n", encoding="utf-8")

    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": "exp_validate",
        "workflow_type": "openclaw_lobster",
        "model_config": {"models": {}},
        "openclaw_lobster_workflow_file": manifest["staged_workflow_file"],
        "profile_training_data": manifest["training_data_path"],
        "predict_trace_data": manifest["training_data_path"],
        "search_axes": ["model", "budget"],
        "search_models": ["qwen35-9b-awq"],
        "search_budgets": [10, 200],
        "profile_models": ["qwen35-9b-awq"],
        "latency_models": ["QuantTrio/Qwen3.5-9B-AWQ"],
        "openclaw_agent_policies": {
            "summarize_each": {
                "required_fields": ["summary"],
                "judge": {"mode": "semantic_llm", "prompt": "GT {ground_truth_field} Pred {predicted_field}"},
            },
            "classify": {"required_fields": ["category"], "judge": {"mode": "strict_exact"}},
            "overview": {
                "required_fields": ["overview_paragraph"],
                "judge": {"mode": "semantic_llm", "prompt": "GT {ground_truth_field} Pred {predicted_field}"},
            },
            "ask_questions": {
                "required_fields": ["question"],
                "judge": {"mode": "semantic_llm", "prompt": "GT {ground_truth_field} Pred {predicted_field}"},
            },
            "draft_replies": {
                "required_fields": ["draft_body"],
                "judge": {"mode": "semantic_llm", "prompt": "GT {ground_truth_field} Pred {predicted_field}"},
            },
        },
        "workflow_loops": [
            {
                "name": "email_loop",
                "count": 2,
                "map_nodes": ["summarize_each", "classify"],
                "reduce_node": "overview",
            },
            {
                "name": "reply_loop",
                "count": 2,
                "map_nodes": ["ask_questions", "draft_replies"],
            },
        ],
        "predict_subagent_score_thresholds": {
            "classify": 0.2,
            "summarize_each": 0.3,
        },
    }

    summary = validate_openclaw_config_payload(cfg)

    assert "overview" in summary["workflow_agents"]
    assert summary["normalized_policies"]["classify"]["mode"] == "strict_exact"


def test_validate_openclaw_config_payload_rejects_unknown_threshold_agent(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(repo_root)
    workflow_path = _write_workflow_root(tmp_path / "source_openclaw")
    manifest_path = stage_openclaw_workspace(str(workflow_path), "exp_validate_bad")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _write_json(Path(manifest["training_data_path"]), _training_payload())
    model_config = repo_root / "configs" / "config.qwen35.local.yaml"
    model_config.parent.mkdir(parents=True, exist_ok=True)
    model_config.write_text("models: {}\n", encoding="utf-8")

    policies = {
        "classify": {"required_fields": ["category"], "judge": {"mode": "strict_exact"}},
    }
    for agent, field in [
        ("summarize_each", "summary"),
        ("overview", "overview_paragraph"),
        ("ask_questions", "question"),
        ("draft_replies", "draft_body"),
    ]:
        policies[agent] = {
            "required_fields": [field],
            "judge": {"mode": "semantic_llm", "prompt": "GT {ground_truth_field} Pred {predicted_field}"},
        }

    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": "exp_validate_bad",
        "workflow_type": "openclaw_lobster",
        "model_config": {"models": {}},
        "openclaw_lobster_workflow_file": manifest["staged_workflow_file"],
        "profile_training_data": manifest["training_data_path"],
        "predict_trace_data": manifest["training_data_path"],
        "search_axes": ["model", "budget"],
        "search_models": ["qwen35-9b-awq"],
        "search_budgets": [10, 200],
        "profile_models": ["qwen35-9b-awq"],
        "latency_models": ["QuantTrio/Qwen3.5-9B-AWQ"],
        "openclaw_agent_policies": policies,
        "predict_subagent_score_thresholds": {"missing": 0.5},
    }

    with pytest.raises(ValueError, match="unknown subagent"):
        validate_openclaw_config_payload(cfg)
