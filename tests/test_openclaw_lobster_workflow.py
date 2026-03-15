from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from workflow_compiler.workflows.dsl_registry import get_workflow_module
from workflow_compiler.workflows.openclaw_lobster.parser import parse_lobster_workflow


def _write_lobster_fixture(path: Path) -> None:
    path.write_text(
        """
name: outlook-past-24h
version: 1
steps:
  - id: fetch_full_bodies
    command: ./bin/outlook_fetch_full_bodies

  - id: summarize_each
    stdin: $fetch_full_bodies.stdout
    command: ./bin/outlook_llm_summarize_each

  - id: classify
    stdin: $fetch_full_bodies.stdout
    command: |
      cat <<'__SUM__'
      $summarize_each.stdout
      __SUM__
      ./bin/outlook_llm_classify

  - id: overview
    stdin: $fetch_full_bodies.stdout
    command: |
      cat <<'__SUM__'
      $summarize_each.stdout
      __SUM__
      cat <<'__CLS__'
      $classify.stdout
      __CLS__
      ./bin/outlook_llm_overview

  - id: ask_questions
    stdin: $fetch_full_bodies.stdout
    command: |
      cat <<'__OVERVIEW__'
      $overview.stdout
      __OVERVIEW__
      ./bin/outlook_llm_questions

  - id: draft_replies
    stdin: $fetch_full_bodies.stdout
    command: |
      cat <<'__OVERVIEW__'
      $overview.stdout
      __OVERVIEW__
      cat <<'__ASK__'
      $ask_questions.stdout
      __ASK__
      ./bin/outlook_llm_draft_replies
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
