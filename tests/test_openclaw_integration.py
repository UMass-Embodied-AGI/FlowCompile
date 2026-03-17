from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from workflow_compiler.integration import openclaw
from workflow_compiler.workflows.openclaw_lobster.parser import parse_lobster_workflow


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


def _write_step_script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _write_workflow_bundle(root: Path) -> Path:
    bundle = root / "outlook-past-24h"
    workflow_path = bundle / "workflow.lobster.yaml"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_dir = bundle / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "summarize_each.schema.json").write_text(
        '{"type":"object","required":["summary"],"properties":{"summary":{"type":"string"}},"additionalProperties":false}\n',
        encoding="utf-8",
    )
    (prompts_dir / "classify.schema.json").write_text(
        '{"type":"object","required":["category"],"properties":{"category":{"type":"string"}},"additionalProperties":false}\n',
        encoding="utf-8",
    )
    (prompts_dir / "overview.schema.json").write_text(
        '{"type":"object","required":["overview_paragraph"],"properties":{"overview_paragraph":{"type":"string"}},"additionalProperties":false}\n',
        encoding="utf-8",
    )
    (prompts_dir / "questions.schema.json").write_text(
        '{"type":"object","required":["question"],"properties":{"question":{"type":"string"}},"additionalProperties":false}\n',
        encoding="utf-8",
    )
    (prompts_dir / "draft_replies.schema.json").write_text(
        '{"type":"object","required":["draft_body"],"properties":{"draft_body":{"type":"string"}},"additionalProperties":false}\n',
        encoding="utf-8",
    )
    _write_step_script(
        bundle / "bin" / "summarize_each.py",
        "\n".join(
            [
                "from lobster_workflow.llm import run_json_batch",
                "from helpers import load_schema",
                "",
                "def main():",
                "    items = []",
                "    for item_id in ('m1', 'm2'):",
                "        items.append({'id': item_id, 'input': {'email_id': item_id}})",
                "    run_json_batch(prompt='summarize', schema=load_schema('summarize_each'), items=items, budget_preset='unlimited', agent='summarize_each', max_tokens=1, timeout_ms=1, concurrency=1)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
    )
    _write_step_script(
        bundle / "bin" / "classify.py",
        "\n".join(
            [
                "from lobster_workflow.llm import run_json_batch",
                "from helpers import load_schema",
                "from lobster_workflow.state import load_manifest_from_stdin, read_artifact",
                "",
                "def main():",
                "    manifest = load_manifest_from_stdin()",
                "    read_artifact(manifest, 'fetch_full_bodies')",
                "    read_artifact(manifest, 'summarize_each')",
                "    items = []",
                "    for item_id in ('m1', 'm2'):",
                "        items.append({'id': item_id, 'input': {'email_id': item_id}})",
                "    run_json_batch(prompt='classify', schema=load_schema('classify'), items=items, budget_preset='unlimited', agent='classify', max_tokens=1, timeout_ms=1, concurrency=1)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
    )
    _write_step_script(
        bundle / "bin" / "overview.py",
        "\n".join(
            [
                "from lobster_workflow.llm import run_json_batch",
                "from helpers import load_schema",
                "from lobster_workflow.state import load_manifest_from_stdin, read_artifact",
                "",
                "def main():",
                "    manifest = load_manifest_from_stdin()",
                "    read_artifact(manifest, 'fetch_full_bodies')",
                "    read_artifact(manifest, 'summarize_each')",
                "    read_artifact(manifest, 'classify')",
                "    items = [{'id': 'combined', 'input': {'kind': 'summary'}}]",
                "    run_json_batch(prompt='overview', schema=load_schema('overview'), items=items, budget_preset='unlimited', agent='overview', max_tokens=1, timeout_ms=1, concurrency=1)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
    )
    _write_step_script(
        bundle / "bin" / "ask_questions.py",
        "\n".join(
            [
                "from lobster_workflow.llm import run_json_batch",
                "from helpers import load_schema",
                "from lobster_workflow.state import load_manifest_from_stdin, read_artifact",
                "",
                "def main():",
                "    manifest = load_manifest_from_stdin()",
                "    read_artifact(manifest, 'fetch_full_bodies')",
                "    read_artifact(manifest, 'overview')",
                "    items = []",
                "    for item_id in ('m1', 'm2'):",
                "        items.append({'id': item_id, 'input': {'email_id': item_id}})",
                "    run_json_batch(prompt='questions', schema=load_schema('questions'), items=items, budget_preset='unlimited', agent='ask_questions', max_tokens=1, timeout_ms=1, concurrency=1)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
    )
    _write_step_script(
        bundle / "bin" / "draft_replies.py",
        "\n".join(
            [
                "from lobster_workflow.llm import run_json_batch",
                "from helpers import load_schema",
                "from lobster_workflow.state import load_manifest_from_stdin, read_artifact",
                "",
                "def main():",
                "    manifest = load_manifest_from_stdin()",
                "    read_artifact(manifest, 'fetch_full_bodies')",
                "    read_artifact(manifest, 'overview')",
                "    read_artifact(manifest, 'ask_questions')",
                "    read_artifact(manifest, 'collect_answers')",
                "    items = []",
                "    for item_id in ('m1', 'm2'):",
                "        items.append({'id': item_id, 'input': {'email_id': item_id}})",
                "    run_json_batch(prompt='drafts', schema=load_schema('draft_replies'), items=items, budget_preset='unlimited', agent='draft_replies', max_tokens=1, timeout_ms=1, concurrency=1)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
    )
    workflow_path.write_text(
        "\n".join(
            [
                "name: outlook-past-24h",
                "version: 1",
                "steps:",
                "  - id: fetch_full_bodies",
                "    command: echo fetch",
                "  - id: summarize_each",
                "    stdin: $fetch_full_bodies.stdout",
                "    command: |",
                '      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/summarize_each.py',
                "  - id: classify",
                "    stdin: $summarize_each.stdout",
                "    command: |",
                '      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/classify.py',
                "  - id: overview",
                "    stdin: $classify.stdout",
                "    command: |",
                '      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/overview.py',
                "  - id: collect_answers",
                "    stdin: $overview.stdout",
                "    command: echo collect",
                "  - id: ask_questions",
                "    stdin: $overview.stdout",
                "    command: |",
                '      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/ask_questions.py',
                "  - id: draft_replies",
                "    stdin: $collect_answers.stdout",
                "    command: |",
                '      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/draft_replies.py',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle


REAL_ARXIV_WORKFLOW = Path("/home/junyan/.openclaw/workspace/workflows/arxiv-cscl-efficiency/workflow.lobster.yaml")
REAL_OUTLOOK_WORKFLOW = Path("/home/junyan/.openclaw/workspace/workflows/outlook-past-24h/workflow.lobster.yaml")


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


def test_parse_lobster_workflow_detects_python_step_llm_nodes(tmp_path: Path):
    bundle = _write_workflow_bundle(tmp_path)

    spec = parse_lobster_workflow(str(bundle / "workflow.lobster.yaml"))
    node_ids = [node["id"] for node in spec["nodes"]]
    edge_pairs = {(edge["from"], edge["to"]) for edge in spec["edges"]}

    assert node_ids == ["summarize_each", "classify", "overview", "ask_questions", "draft_replies"]
    assert ("summarize_each", "classify") in edge_pairs
    assert ("classify", "overview") in edge_pairs
    assert ("summarize_each", "overview") in edge_pairs
    assert ("ask_questions", "draft_replies") in edge_pairs


@pytest.mark.skipif(not REAL_ARXIV_WORKFLOW.exists(), reason="real arxiv workflow not available")
def test_parse_lobster_workflow_real_arxiv_bundle() -> None:
    spec = parse_lobster_workflow(str(REAL_ARXIV_WORKFLOW))
    node_ids = [node["id"] for node in spec["nodes"]]

    assert node_ids == ["select_efficiency", "summarize_pdfs"]


@pytest.mark.skipif(not REAL_OUTLOOK_WORKFLOW.exists(), reason="real outlook workflow not available")
def test_parse_lobster_workflow_real_outlook_bundle() -> None:
    spec = parse_lobster_workflow(str(REAL_OUTLOOK_WORKFLOW))
    node_ids = [node["id"] for node in spec["nodes"]]
    edge_pairs = {(edge["from"], edge["to"]) for edge in spec["edges"]}

    assert node_ids == ["summarize_each", "classify", "overview", "ask_questions", "draft_replies"]
    assert ("summarize_each", "overview") in edge_pairs
    assert ("classify", "overview") in edge_pairs
    assert ("ask_questions", "draft_replies") in edge_pairs


@pytest.mark.parametrize(
    ("workflow_path", "expected_step_id"),
    [
        pytest.param(REAL_ARXIV_WORKFLOW, "select_efficiency", id="arxiv"),
        pytest.param(REAL_OUTLOOK_WORKFLOW, "summarize_each", id="outlook"),
    ],
)
def test_demo_run_openclaw_real_bundle_gets_past_parser_validation(
    monkeypatch,
    workflow_path: Path,
    expected_step_id: str,
) -> None:
    if not workflow_path.exists():
        pytest.skip("real workflow not available")

    bundle = workflow_path.parent

    def fake_build_runtime_env(session_path: Path, capture_path: Path, env_overrides: dict[str, str]) -> dict[str, str]:
        return {
            "FLOWCOMPILE_OPENCLAW_CAPTURE_FILE": str(capture_path),
            **env_overrides,
        }

    def fake_run_lobster_payload(cmd, cwd: Path, env: dict[str, str]) -> dict:
        assert cwd == bundle.resolve()
        instrumented_path = Path(cmd[cmd.index("--file") + 1])
        assert expected_step_id in instrumented_path.read_text(encoding="utf-8")
        return {"ok": True, "status": "needs_approval", "requiresApproval": {"resumeToken": "tok", "items": []}}

    monkeypatch.setattr(openclaw, "_build_runtime_env", fake_build_runtime_env)
    monkeypatch.setattr(openclaw, "_run_lobster_payload", fake_run_lobster_payload)

    session_path = openclaw.demo_run_openclaw(str(bundle))
    session = json.loads(Path(session_path).read_text(encoding="utf-8"))

    assert session["status"] == "needs_approval"
    assert session["resume_token"] == "tok"


def test_run_json_batch_capture_records_round_trip(monkeypatch, tmp_path: Path):
    workspace_lib = Path("/home/junyan/.openclaw/workspace/workflows/_lib/python")
    if str(workspace_lib) not in sys.path:
        sys.path.insert(0, str(workspace_lib))
    from lobster_workflow import llm as workflow_llm

    capture_path = tmp_path / "capture.jsonl"
    monkeypatch.setenv("FLOWCOMPILE_OPENCLAW_CAPTURE_FILE", str(capture_path))
    monkeypatch.setenv("FLOWCOMPILE_OPENCLAW_STEP_ID", "summarize_each")

    async def fake_run_batch(**_: object) -> list[dict[str, object]]:
        return [
            {"id": "e1", "result": {"summary": "s1"}, "error": ""},
            {"id": "e2", "result": None, "error": "boom"},
        ]

    monkeypatch.setattr(workflow_llm, "_run_batch", fake_run_batch)

    results = workflow_llm.run_json_batch(
        prompt="summarize",
        schema={"required": ["summary"]},
        items=[
            {"id": "e1", "input": {"email": {"id": "e1"}}},
            {"id": "e2", "input": {"email": {"id": "e2"}}},
        ],
        budget_preset="unlimited",
        agent="summarize_each",
        max_tokens=10,
        timeout_ms=50,
        concurrency=2,
    )

    assert len(results) == 2
    records = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["step_id"] == "summarize_each"
    assert records[0]["request_args"]["input"] == {"email": {"id": "e1"}}
    assert records[0]["details_json"] == {"summary": "s1"}
    assert records[1]["error"] == "boom"

    training = openclaw._build_training_data(records, run_label="demo")
    assert [sample["agent_name"] for sample in training] == ["summarize_each"]
    assert training[0]["processed_output"] == '{"summary":"s1"}'


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
        instrumented_path = Path(cmd[cmd.index("--file") + 1])
        instrumented_text = instrumented_path.read_text(encoding="utf-8")
        assert "FLOWCOMPILE_OPENCLAW_STEP_ID=\"summarize_each\"" in instrumented_text
        assert "FLOWCOMPILE_OPENCLAW_STEP_ID=\"overview\"" in instrumented_text
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
    assert session["instrumented_workflow_file"].endswith("instrumented_workflow.lobster.yaml")
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
    assert analysis["agents"]["ask_questions"]["resolved_schema_path"].endswith("questions.schema.json")
    assert analysis["agents"]["overview"]["schema_property_types"]["overview_paragraph"] == ["string"]
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
    assert summary["agent_schema_paths"]["ask_questions"].endswith("questions.schema.json")


def test_validate_openclaw_config_payload_rejects_required_fields_missing_from_schema(tmp_path: Path):
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
        "search_budgets": [10],
        "profile_models": ["qwen35-9b-awq"],
        "latency_models": ["QuantTrio/Qwen3.5-9B-AWQ"],
        "openclaw_agent_policies": {
            "summarize_each": {
                "required_fields": ["missing_field"],
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

    with pytest.raises(ValueError, match="unknown schema properties"):
        openclaw.validate_openclaw_config_payload(cfg, config_path=str(config_path))


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
