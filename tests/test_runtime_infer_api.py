from __future__ import annotations

from pathlib import Path

import pytest

import workflow_compiler.runtime.infer as runtime_infer


def _compiled_config() -> dict:
    return {
        "config_id": "cfg_0000",
        "structure_id": "full",
        "metrics": {"expected_accuracy": 0.9, "expected_latency": 1.0},
        "agents": {},
    }


def _constraint_configs() -> list[dict]:
    return [
        {
            "config_id": "cfg_a",
            "metrics": {"expected_accuracy": 0.40, "expected_latency": 1.0},
        },
        {
            "config_id": "cfg_b",
            "metrics": {"expected_accuracy": 0.55, "expected_latency": 2.0},
        },
        {
            "config_id": "cfg_c",
            "metrics": {"expected_accuracy": 0.70, "expected_latency": 3.0},
        },
        {
            "config_id": "cfg_d",
            "metrics": {"expected_accuracy": 0.90, "expected_latency": 4.0},
        },
    ]


def test_infer_runtime_returns_selected_config_and_answer(monkeypatch, tmp_path: Path):
    def fake_run_batch_sync(pairs, workflow_type, output_dir):
        assert workflow_type == "math"
        assert output_dir == tmp_path
        assert len(pairs) == 1
        query, config = pairs[0]
        assert query["problem"] == "Solve 1+1"
        assert config["config_id"] == "cfg_0000"
        return [
            {
                "query_id": "q1",
                "output": "2",
                "structure_id": "full",
                "config_id": "cfg_0000",
                "output_dir": "runtime_outputs/q1",
            }
        ]

    monkeypatch.setattr(runtime_infer, "run_batch_sync", fake_run_batch_sync)

    result = runtime_infer.infer_runtime(
        query="Solve 1+1",
        configs=[_compiled_config()],
        workflow_type="math",
        output_dir=tmp_path,
        query_id="q1",
    )

    assert result["answer"] == "2"
    assert result["selected_config"]["config_id"] == "cfg_0000"


def test_select_runtime_config_raises_for_no_match():
    with pytest.raises(SystemExit, match="No runtime config matched"):
        runtime_infer.select_runtime_config([], strategy="preference")


def test_select_runtime_config_constraint_prefers_lowest_accuracy_meeting_min():
    selected = runtime_infer.select_runtime_config(
        _constraint_configs(),
        strategy="constraint",
        min_accuracy=0.50,
    )
    assert selected["config_id"] == "cfg_b"


def test_select_runtime_config_constraint_prefers_highest_latency_meeting_max():
    selected = runtime_infer.select_runtime_config(
        _constraint_configs(),
        strategy="constraint",
        max_latency=3.2,
    )
    assert selected["config_id"] == "cfg_c"


def test_select_runtime_config_constraint_with_both_uses_boundary_rule():
    selected = runtime_infer.select_runtime_config(
        _constraint_configs(),
        strategy="constraint",
        min_accuracy=0.60,
        max_latency=3.2,
    )
    assert selected["config_id"] == "cfg_c"
