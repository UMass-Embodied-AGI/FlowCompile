from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow_compiler.integration import openclaw


def _openclaw_model_config_text(*, include_semantic_judge_model: bool = True) -> str:
    lines = [
        "endpoints:",
        '  local_base_url: "http://127.0.0.1:4000"',
        '  profile_base_url: "http://profile-host:4000"',
        "models:",
        "  qwen35-9b-awq:",
        '    api_type: "openai"',
        '    api_key: "dummy"',
        '    hf_model_name: "QuantTrio/Qwen3.5-9B-AWQ"',
    ]
    if include_semantic_judge_model:
        lines.extend(
            [
                "  gpt-oss-120b:",
                '    api_type: "openai"',
                '    api_key: "dummy"',
                '    hf_model_name: "openai/gpt-oss-120b"',
            ]
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_workflow_bundle(root: Path) -> Path:
    bundle = root / "outlook-past-24h"
    workflow_path = bundle / "workflow.lobster.yaml"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(
        "\n".join(
            [
                "name: outlook-past-24h",
                "version: 1",
                "steps:",
                "  - id: summarize_each",
                "    command: ./bin/outlook_llm_summarize_each",
                "  - id: classify",
                "    stdin: $summarize_each.stdout",
                "    command: ./bin/outlook_llm_classify",
                "  - id: overview",
                "    command: |",
                "      cat <<'__SUM__'",
                "      $summarize_each.stdout",
                "      __SUM__",
                "      cat <<'__CLS__'",
                "      $classify.stdout",
                "      __CLS__",
                "      ./bin/outlook_llm_overview",
                "  - id: ask_questions",
                "    stdin: $overview.stdout",
                "    command: ./bin/outlook_llm_questions",
                "  - id: draft_replies",
                "    stdin: $ask_questions.stdout",
                "    command: ./bin/outlook_llm_draft_replies",
            ]
        ),
        encoding="utf-8",
    )
    return bundle


def _capture_records() -> list[dict]:
    return [
        {
            "timestamp": "2026-03-15T12:00:00-05:00",
            "step_id": "summarize_each",
            "request_args": {
                "prompt": "summarize prompt",
                "input": {"email_id": "e1"},
                "schema": {"required": ["summary"]},
            },
            "details_json": {"summary": "s1"},
        },
        {
            "timestamp": "2026-03-15T12:00:01-05:00",
            "step_id": "classify",
            "request_args": {
                "prompt": "classify prompt",
                "input": {"email_id": "e1"},
                "schema": {"required": ["category"]},
            },
            "details_json": {"category": "reply_required"},
        },
    ]


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


def _training_payload_single_sample_per_agent() -> dict:
    return {
        "training_data": [
            {
                "agent_name": "summarize_each",
                "raw_llm_prompt": "summarize prompt",
                "processed_output": '{"summary":"s1"}',
                "raw_llm_output": '{"summary":"s1"}',
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
                "agent_name": "draft_replies",
                "raw_llm_prompt": "draft prompt",
                "processed_output": '{"draft_body":"body1"}',
                "raw_llm_output": '{"draft_body":"body1"}',
            },
        ]
    }


def _training_payload_missing_agents() -> dict:
    return {
        "training_data": [
            {
                "agent_name": "summarize_each",
                "raw_llm_prompt": "summarize prompt",
                "processed_output": '{"summary":"s1"}',
                "raw_llm_output": '{"summary":"s1"}',
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
        ]
    }


def test_demo_run_openclaw_creates_bundle_local_artifacts(monkeypatch, tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)

    def fake_build_runtime_env(session_path: Path, capture_path: Path, env_overrides: dict[str, str]) -> dict[str, str]:
        return {
            "FLOWCOMPILE_OPENCLAW_CAPTURE_FILE": str(capture_path),
            **env_overrides,
        }

    def fake_run_lobster_payload(cmd, cwd: Path, env: dict[str, str]) -> dict:
        assert cwd == bundle.resolve()
        assert cmd[:4] == ["lobster", "run", "--mode", "tool"]
        assert "--file" in cmd
        _append_jsonl(Path(env["FLOWCOMPILE_OPENCLAW_CAPTURE_FILE"]), _capture_records())
        return {"ok": True, "status": "ok"}

    monkeypatch.setattr(openclaw, "_build_runtime_env", fake_build_runtime_env)
    monkeypatch.setattr(openclaw, "_run_lobster_payload", fake_run_lobster_payload)

    session_path = openclaw.demo_run_openclaw(str(bundle), args_json='{"client_id":"demo"}')

    manifest = json.loads((bundle / "flowcompile" / "manifest.json").read_text(encoding="utf-8"))
    session = json.loads(Path(session_path).read_text(encoding="utf-8"))
    training = json.loads((bundle / "flowcompile" / "flowcompile_training.json").read_text(encoding="utf-8"))

    assert manifest["workflow_dir"] == str(bundle.resolve())
    assert manifest["workflow_file"] == str((bundle / "workflow.lobster.yaml").resolve())
    assert Path(manifest["session_path"]) == bundle / "flowcompile" / "session" / "session.json"
    assert session["status"] == "completed"
    assert session["llm_call_count"] == 2
    assert training["metadata"]["workflow"] == str((bundle / "workflow.lobster.yaml").resolve())
    assert len(training["training_data"]) == 2


def test_demo_resume_openclaw_updates_pause_state(monkeypatch, tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)
    manifest_path, manifest = openclaw._prepare_manifest(str(bundle))
    _write_json(
        Path(manifest["session_path"]),
        {
            "schema_version": openclaw.SESSION_SCHEMA_VERSION,
            "manifest_path": str(manifest_path),
            "capture_path": manifest["capture_path"],
            "status": "needs_approval",
            "env_overrides": {"EXISTING": "1"},
            "resume_token": "resume-123",
        },
    )

    def fake_build_runtime_env(session_path: Path, capture_path: Path, env_overrides: dict[str, str]) -> dict[str, str]:
        assert env_overrides["EXISTING"] == "1"
        return env_overrides

    def fake_run_lobster_payload(cmd, cwd: Path, env: dict[str, str]) -> dict:
        assert cwd == bundle.resolve()
        assert cmd == ["lobster", "resume", "--mode", "tool", "--token", "resume-123", "--approve", "yes"]
        return {
            "ok": True,
            "status": "needs_approval",
            "requiresApproval": {
                "resumeToken": "resume-456",
                "items": [{"title": "question"}],
                "preview": "preview text",
            },
        }

    monkeypatch.setattr(openclaw, "_build_runtime_env", fake_build_runtime_env)
    monkeypatch.setattr(openclaw, "_run_lobster_payload", fake_run_lobster_payload)

    session_path = openclaw.demo_resume_openclaw(str(bundle))
    session = json.loads(Path(session_path).read_text(encoding="utf-8"))

    assert session["status"] == "needs_approval"
    assert session["resume_token"] == "resume-456"
    assert session["pause_metadata"]["item_count"] == 1
    assert session["last_approval_payload"]["preview"] == "preview text"


def test_analyze_openclaw_demo_emits_bundle_local_relative_paths(tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)
    manifest_path, manifest = openclaw._prepare_manifest(str(bundle))
    _write_json(Path(manifest["training_data_path"]), _training_payload())

    analysis_path = openclaw.analyze_openclaw_demo(str(bundle))
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))

    assert analysis["training_data_summary"]["counts_by_agent"]["summarize_each"] == 2
    loops = analysis["candidate_workflow_loops"]
    assert any(
        loop["reduce_node"] == "overview"
        and loop["inference_source"] == "structure"
        and loop["requires_human_confirmation"] is True
        for loop in loops
        if "reduce_node" in loop
    )
    assert any(
        loop["map_nodes"] == ["ask_questions", "draft_replies"]
        and loop["count"] == 2
        and loop["count_source"] == "observed_demo_hint"
        for loop in loops
    )
    assert analysis["agents"]["overview"]["required_fields_intersection"] == ["overview_paragraph"]
    assert analysis["config_authoring"]["relative_paths"]["openclaw_lobster_workflow_file"] == "../workflow.lobster.yaml"
    assert analysis["config_authoring"]["relative_paths"]["profile_training_data"] == "flowcompile_training.json"
    assert analysis["config_authoring"]["default_values"]["experiment_root"] == "."
    assert "model_config" not in analysis["config_authoring"]["relative_paths"]
    assert analysis["manifest_path"] == str(manifest_path.resolve())


def test_analyze_openclaw_demo_rejects_missing_llm_steps(tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)
    manifest_path, manifest = openclaw._prepare_manifest(str(bundle))
    _write_json(Path(manifest["training_data_path"]), _training_payload_missing_agents())

    with pytest.raises(ValueError, match="missing captured samples.*ask_questions, draft_replies"):
        openclaw.analyze_openclaw_demo(str(bundle))

    assert manifest_path.exists()
    assert not Path(manifest["analysis_path"]).exists()


def test_analyze_openclaw_demo_emits_structural_loops_even_for_single_item_counts(tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)
    manifest_path, manifest = openclaw._prepare_manifest(str(bundle))
    _write_json(Path(manifest["training_data_path"]), _training_payload_single_sample_per_agent())

    analysis_path = openclaw.analyze_openclaw_demo(str(bundle))
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))

    loops = analysis["candidate_workflow_loops"]
    assert any(
        loop["map_nodes"] == ["summarize_each", "classify"]
        and loop.get("reduce_node") == "overview"
        and loop["count"] == 1
        and loop["inference_source"] == "structure"
        for loop in loops
    )
    assert any(
        loop["map_nodes"] == ["ask_questions", "draft_replies"]
        and "reduce_node" not in loop
        and loop["count"] == 1
        and loop["inference_source"] == "structure"
        for loop in loops
    )
    assert analysis["manifest_path"] == str(manifest_path.resolve())


def test_validate_openclaw_config_payload_accepts_config_relative_bundle_paths(tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)
    flowcompile_dir = bundle / "flowcompile"
    training_path = flowcompile_dir / "flowcompile_training.json"
    model_config = flowcompile_dir / "model-config.yaml"
    config_path = flowcompile_dir / "flowcompile_openclaw.yaml"
    _write_json(training_path, _training_payload())
    model_config.write_text(_openclaw_model_config_text(), encoding="utf-8")

    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": bundle.name,
        "experiment_root": ".",
        "workflow_type": "openclaw_lobster",
        "model_config": "model-config.yaml",
        "openclaw_lobster_workflow_file": "../workflow.lobster.yaml",
        "profile_training_data": "flowcompile_training.json",
        "predict_trace_data": "flowcompile_training.json",
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

    summary = openclaw.validate_openclaw_config_payload(cfg, config_path=str(config_path))

    assert "overview" in summary["workflow_agents"]
    assert summary["normalized_policies"]["classify"]["mode"] == "strict_exact"


def test_validate_openclaw_config_payload_rejects_missing_semantic_judge_model(tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)
    flowcompile_dir = bundle / "flowcompile"
    training_path = flowcompile_dir / "flowcompile_training.json"
    model_config = flowcompile_dir / "model-config.yaml"
    config_path = flowcompile_dir / "flowcompile_openclaw.yaml"
    _write_json(training_path, _training_payload())
    model_config.write_text(_openclaw_model_config_text(include_semantic_judge_model=False), encoding="utf-8")

    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": bundle.name,
        "experiment_root": ".",
        "workflow_type": "openclaw_lobster",
        "model_config": "model-config.yaml",
        "openclaw_lobster_workflow_file": "../workflow.lobster.yaml",
        "profile_training_data": "flowcompile_training.json",
        "predict_trace_data": "flowcompile_training.json",
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
    }

    with pytest.raises(ValueError, match="gpt-oss-120b"):
        openclaw.validate_openclaw_config_payload(cfg, config_path=str(config_path))


def test_validate_openclaw_config_payload_allows_missing_semantic_judge_model_for_strict_exact_only(tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)
    flowcompile_dir = bundle / "flowcompile"
    training_path = flowcompile_dir / "flowcompile_training.json"
    model_config = flowcompile_dir / "model-config.yaml"
    config_path = flowcompile_dir / "flowcompile_openclaw.yaml"
    _write_json(training_path, _training_payload())
    model_config.write_text(_openclaw_model_config_text(include_semantic_judge_model=False), encoding="utf-8")

    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": bundle.name,
        "experiment_root": ".",
        "workflow_type": "openclaw_lobster",
        "model_config": "model-config.yaml",
        "openclaw_lobster_workflow_file": "../workflow.lobster.yaml",
        "profile_training_data": "flowcompile_training.json",
        "predict_trace_data": "flowcompile_training.json",
        "search_axes": ["model", "budget"],
        "search_models": ["qwen35-9b-awq"],
        "search_budgets": [10, 200],
        "profile_models": ["qwen35-9b-awq"],
        "latency_models": ["QuantTrio/Qwen3.5-9B-AWQ"],
        "openclaw_agent_policies": {
            "summarize_each": {"required_fields": ["summary"], "judge": {"mode": "strict_exact"}},
            "classify": {"required_fields": ["category"], "judge": {"mode": "strict_exact"}},
            "overview": {"required_fields": ["overview_paragraph"], "judge": {"mode": "strict_exact"}},
            "ask_questions": {"required_fields": ["question"], "judge": {"mode": "strict_exact"}},
            "draft_replies": {"required_fields": ["draft_body"], "judge": {"mode": "strict_exact"}},
        },
    }

    summary = openclaw.validate_openclaw_config_payload(cfg, config_path=str(config_path))

    assert "overview" in summary["workflow_agents"]


def test_validate_openclaw_config_payload_rejects_unknown_threshold_agent(tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)
    flowcompile_dir = bundle / "flowcompile"
    training_path = flowcompile_dir / "flowcompile_training.json"
    model_config = flowcompile_dir / "model-config.yaml"
    config_path = flowcompile_dir / "flowcompile_openclaw.yaml"
    _write_json(training_path, _training_payload())
    model_config.write_text(_openclaw_model_config_text(), encoding="utf-8")

    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": bundle.name,
        "experiment_root": ".",
        "workflow_type": "openclaw_lobster",
        "model_config": "model-config.yaml",
        "openclaw_lobster_workflow_file": "../workflow.lobster.yaml",
        "profile_training_data": "flowcompile_training.json",
        "predict_trace_data": "flowcompile_training.json",
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
        "predict_subagent_score_thresholds": {"missing": 0.5},
    }

    with pytest.raises(ValueError, match="unknown subagent"):
        openclaw.validate_openclaw_config_payload(cfg, config_path=str(config_path))
