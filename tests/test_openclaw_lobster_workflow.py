from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from workflow_compiler.workflows.dsl_registry import get_workflow_module
from workflow_compiler.workflows.openclaw_lobster.parser import parse_lobster_workflow


def _write_lobster_fixture(path: Path) -> None:
    bundle_dir = path.parent
    bin_dir = bundle_dir / "bin"
    prompts_dir = bundle_dir / "prompts"
    bin_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    def write_script(name: str, body: str) -> None:
        (bin_dir / name).write_text(body, encoding="utf-8")

    def write_schema(name: str, payload: str) -> None:
        (prompts_dir / f"{name}.schema.json").write_text(payload, encoding="utf-8")

    write_schema("summarize_each", '{"type":"object","required":["summary"],"properties":{"summary":{"type":"string"}},"additionalProperties":false}\n')
    write_schema("classify", '{"type":"object","required":["category"],"properties":{"category":{"type":"string"}},"additionalProperties":false}\n')
    write_schema("overview", '{"type":"object","required":["overview_paragraph"],"properties":{"overview_paragraph":{"type":"string"}},"additionalProperties":false}\n')
    write_schema("questions", '{"type":"object","required":["question"],"properties":{"question":{"type":"string"}},"additionalProperties":false}\n')
    write_schema("draft_replies", '{"type":"object","required":["draft_body"],"properties":{"draft_body":{"type":"string"}},"additionalProperties":false}\n')

    write_script(
        "summarize_each.py",
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
    write_script(
        "classify.py",
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
    write_script(
        "overview.py",
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
    write_script(
        "ask_questions.py",
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
    write_script(
        "draft_replies.py",
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

    path.write_text(
        """
name: outlook-past-24h
version: 1
steps:
  - id: fetch_full_bodies
    command: echo fetch

  - id: summarize_each
    stdin: $fetch_full_bodies.stdout
    command: |
      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/summarize_each.py

  - id: classify
    stdin: $summarize_each.stdout
    command: |
      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/classify.py

  - id: overview
    stdin: $classify.stdout
    command: |
      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/overview.py

  - id: ask_questions
    stdin: $overview.stdout
    command: |
      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/ask_questions.py

  - id: draft_replies
    stdin: $ask_questions.stdout
    command: |
      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/draft_replies.py
""".strip(),
        encoding="utf-8",
    )


def _tiny_df(accuracy: float, latency: float) -> pd.DataFrame:
    return pd.DataFrame({"setting": ["s0"], "accuracy": [accuracy], "latency": [latency]})


def test_parse_lobster_workflow_builds_expected_operator_annotations(tmp_path: Path):
    workflow_file = tmp_path / "outlook.lobster.yaml"
    _write_lobster_fixture(workflow_file)

    spec = parse_lobster_workflow(str(workflow_file))
    node_by_id = {node["id"]: node for node in spec["nodes"]}
    assert set(node_by_id.keys()) == {
        "summarize_each",
        "classify",
        "overview",
        "ask_questions",
        "draft_replies",
    }

    assert node_by_id["summarize_each"]["metadata"]["operator"] == "map"
    assert node_by_id["classify"]["metadata"]["operator"] == "map"
    assert node_by_id["overview"]["metadata"]["operator"] == "map_reduce"
    assert node_by_id["ask_questions"]["metadata"]["operator"] == "map"
    assert node_by_id["draft_replies"]["metadata"]["operator"] == "map"
    assert node_by_id["summarize_each"]["metadata"]["output_schema_path"].endswith("summarize_each.schema.json")
    assert node_by_id["classify"]["metadata"]["output_schema_path"].endswith("classify.schema.json")
    assert node_by_id["overview"]["metadata"]["output_schema_path"].endswith("overview.schema.json")
    assert node_by_id["ask_questions"]["metadata"]["output_schema_path"].endswith("questions.schema.json")
    assert node_by_id["draft_replies"]["metadata"]["output_schema_path"].endswith("draft_replies.schema.json")

    overview_inputs = node_by_id["overview"]["io"]["inputs"]["items"]
    overview_refs = [item["ref"] for item in overview_inputs]
    assert overview_refs == ["state.summarize_each", "state.classify"]
    assert node_by_id["draft_replies"]["io"]["inputs"]["source"]["ref"] == "state.ask_questions"

    edges = {(edge["from"], edge["to"]) for edge in spec["edges"]}
    assert ("summarize_each", "classify") in edges
    assert ("classify", "overview") in edges
    assert ("summarize_each", "overview") in edges
    assert ("overview", "ask_questions") in edges
    assert ("ask_questions", "draft_replies") in edges


def test_parse_lobster_workflow_infers_singleton_reduce_without_name_hints(tmp_path: Path):
    workflow_file = tmp_path / "generic.lobster.yaml"
    bundle_dir = workflow_file.parent
    bin_dir = bundle_dir / "bin"
    prompts_dir = bundle_dir / "prompts"
    bin_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "alpha.schema.json").write_text(
        '{"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}},"additionalProperties":false}\n',
        encoding="utf-8",
    )
    (prompts_dir / "beta.schema.json").write_text(
        '{"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}},"additionalProperties":false}\n',
        encoding="utf-8",
    )

    (bin_dir / "alpha.py").write_text(
        "\n".join(
            [
                "from lobster_workflow.llm import run_json_batch",
                "import json",
                "from pathlib import Path",
                "",
                "SCHEMA_FILE = Path(__file__).resolve().parent.parent / 'prompts' / 'alpha.schema.json'",
                "",
                "def main():",
                "    items = []",
                "    for item_id in ('a1', 'a2'):",
                "        items.append({'id': item_id, 'input': {'value': item_id}})",
                "    schema = json.loads(SCHEMA_FILE.read_text(encoding='utf-8'))",
                "    run_json_batch(prompt='alpha', schema=schema, items=items, budget_preset='unlimited', agent='alpha', max_tokens=1, timeout_ms=1, concurrency=1)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (bin_dir / "beta.py").write_text(
        "\n".join(
            [
                "from lobster_workflow.llm import run_json_batch",
                "from helpers import load_schema",
                "from lobster_workflow.state import load_manifest_from_stdin, read_artifact",
                "",
                "def main():",
                "    manifest = load_manifest_from_stdin()",
                "    read_artifact(manifest, 'alpha')",
                "    items = [{'id': 'once', 'input': {'mode': 'single'}}]",
                "    run_json_batch(prompt='beta', schema=load_schema('beta'), items=items, budget_preset='unlimited', agent='beta', max_tokens=1, timeout_ms=1, concurrency=1)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    workflow_file.write_text(
        "\n".join(
            [
                "name: generic",
                "version: 1",
                "steps:",
                "  - id: alpha",
                "    command: |",
                '      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/alpha.py',
                "  - id: beta",
                "    stdin: $alpha.stdout",
                "    command: |",
                '      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/beta.py',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spec = parse_lobster_workflow(str(workflow_file))
    node_by_id = {node["id"]: node for node in spec["nodes"]}

    assert node_by_id["alpha"]["metadata"]["operator"] == "map"
    assert node_by_id["beta"]["metadata"]["operator"] == "reduce"
    assert node_by_id["beta"]["io"]["inputs"]["items"] == [{"ref": "state.alpha"}]
    assert node_by_id["alpha"]["metadata"]["output_schema_path"].endswith("alpha.schema.json")
    assert node_by_id["beta"]["metadata"]["output_schema_path"].endswith("beta.schema.json")


def test_parse_lobster_workflow_rejects_dynamic_schema_resolution(tmp_path: Path):
    workflow_file = tmp_path / "dynamic.lobster.yaml"
    bundle_dir = workflow_file.parent
    bin_dir = bundle_dir / "bin"
    prompts_dir = bundle_dir / "prompts"
    bin_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "alpha.schema.json").write_text(
        '{"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}},"additionalProperties":false}\n',
        encoding="utf-8",
    )
    (bin_dir / "alpha.py").write_text(
        "\n".join(
            [
                "from lobster_workflow.llm import run_json_batch",
                "",
                "def choose_schema_name(kind):",
                "    return kind",
                "",
                "def main():",
                "    schema_name = choose_schema_name('alpha')",
                "    run_json_batch(prompt='alpha', schema=load_schema(schema_name), items=[{'id': 'a1', 'input': {}}], budget_preset='unlimited', agent='alpha', max_tokens=1, timeout_ms=1, concurrency=1)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    workflow_file.write_text(
        "\n".join(
            [
                "name: dynamic",
                "version: 1",
                "steps:",
                "  - id: alpha",
                "    command: |",
                '      "${LOBSTER_WORKFLOW_PYTHON:-python3}" ./bin/alpha.py',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="statically discoverable schema file"):
        parse_lobster_workflow(str(workflow_file))


def test_openclaw_lobster_backward_map_reduce_formula(tmp_path: Path):
    workflow_file = tmp_path / "outlook.lobster.yaml"
    _write_lobster_fixture(workflow_file)

    workflow = get_workflow_module(
        "openclaw_lobster",
        openclaw_lobster_workflow_file=str(workflow_file),
    )
    structure = workflow.get_full_structure()
    payload = {
        "structure": structure,
        "metrics": {
            "summarize_each": _tiny_df(0.8, 1.0),
            "classify": _tiny_df(0.7, 2.0),
            "overview": _tiny_df(0.9, 3.0),
            "ask_questions": _tiny_df(0.6, 4.0),
            "draft_replies": _tiny_df(0.5, 5.0),
        },
    }
    result = workflow.backward(payload)
    assert len(result) == 1
    expected_accuracy = (0.8 * 0.7) * 0.9 * (0.6 * 0.5)
    assert float(result["workflow_accuracy"].iloc[0]) == pytest.approx(expected_accuracy)
    assert float(result["workflow_latency"].iloc[0]) == pytest.approx(1.0 + 2.0 + 3.0 + 4.0 + 5.0)


def test_workflow_loops_scale_map_and_count_reduce_once(tmp_path: Path):
    workflow_file = tmp_path / "outlook.lobster.yaml"
    _write_lobster_fixture(workflow_file)

    workflow = get_workflow_module(
        "openclaw_lobster",
        openclaw_lobster_workflow_file=str(workflow_file),
    )
    structure = workflow.get_full_structure()
    payload = {
        "structure": structure,
        "metrics": {
            "summarize_each": _tiny_df(0.8, 1.0),
            "classify": _tiny_df(0.7, 2.0),
            "overview": _tiny_df(0.9, 3.0),
            "ask_questions": _tiny_df(0.6, 4.0),
            "draft_replies": _tiny_df(0.5, 5.0),
        },
        "metadata": {
            "workflow_loops": [
                {
                    "name": "email_loop",
                    "count": 20,
                    "map_nodes": ["summarize_each", "classify"],
                    "reduce_node": "overview",
                },
                {
                    "name": "reply_loop",
                    "count": 2,
                    "map_nodes": ["ask_questions", "draft_replies"],
                },
            ]
        },
    }
    result = workflow.backward(payload)
    assert len(result) == 1
    assert float(result["workflow_accuracy"].iloc[0]) == pytest.approx((0.8 * 0.7) * 0.9 * (0.6 * 0.5))
    assert float(result["workflow_latency"].iloc[0]) == pytest.approx(20.0 * (1.0 + 2.0) + 3.0 + 2.0 * (4.0 + 5.0))


def test_workflow_loops_reject_invalid_reduce_operator(tmp_path: Path):
    workflow_file = tmp_path / "outlook.lobster.yaml"
    _write_lobster_fixture(workflow_file)

    workflow = get_workflow_module(
        "openclaw_lobster",
        openclaw_lobster_workflow_file=str(workflow_file),
    )
    structure = workflow.get_full_structure()
    payload = {
        "structure": structure,
        "metrics": {
            "summarize_each": _tiny_df(0.8, 1.0),
            "classify": _tiny_df(0.7, 2.0),
            "overview": _tiny_df(0.9, 3.0),
            "ask_questions": _tiny_df(0.6, 4.0),
            "draft_replies": _tiny_df(0.5, 5.0),
        },
        "metadata": {
            "workflow_loops": [
                {
                    "name": "bad_reduce",
                    "count": 3,
                    "map_nodes": ["summarize_each"],
                    "reduce_node": "classify",
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="must use operator map_reduce or reduce"):
        workflow.backward(payload)


def test_workflow_loops_reject_overlapping_nodes(tmp_path: Path):
    workflow_file = tmp_path / "outlook.lobster.yaml"
    _write_lobster_fixture(workflow_file)

    workflow = get_workflow_module(
        "openclaw_lobster",
        openclaw_lobster_workflow_file=str(workflow_file),
    )
    structure = workflow.get_full_structure()
    payload = {
        "structure": structure,
        "metrics": {
            "summarize_each": _tiny_df(0.8, 1.0),
            "classify": _tiny_df(0.7, 2.0),
            "overview": _tiny_df(0.9, 3.0),
            "ask_questions": _tiny_df(0.6, 4.0),
            "draft_replies": _tiny_df(0.5, 5.0),
        },
        "metadata": {
            "workflow_loops": [
                {
                    "name": "loop_a",
                    "count": 20,
                    "map_nodes": ["summarize_each", "classify"],
                    "reduce_node": "overview",
                },
                {
                    "name": "loop_b",
                    "count": 2,
                    "map_nodes": ["classify", "ask_questions"],
                },
            ]
        },
    }

    with pytest.raises(ValueError, match="assigned to both"):
        workflow.backward(payload)


def test_workflow_loops_reject_unknown_nodes(tmp_path: Path):
    workflow_file = tmp_path / "outlook.lobster.yaml"
    _write_lobster_fixture(workflow_file)

    workflow = get_workflow_module(
        "openclaw_lobster",
        openclaw_lobster_workflow_file=str(workflow_file),
    )
    structure = workflow.get_full_structure()
    payload = {
        "structure": structure,
        "metrics": {
            "summarize_each": _tiny_df(0.8, 1.0),
            "classify": _tiny_df(0.7, 2.0),
            "overview": _tiny_df(0.9, 3.0),
            "ask_questions": _tiny_df(0.6, 4.0),
            "draft_replies": _tiny_df(0.5, 5.0),
        },
        "metadata": {
            "workflow_loops": [
                {
                    "name": "bad_node",
                    "count": 2,
                    "map_nodes": ["missing_node"],
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="unknown or inactive map node"):
        workflow.backward(payload)
