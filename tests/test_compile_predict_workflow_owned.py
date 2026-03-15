from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from workflow_compiler.compiler import pipeline
from workflow_compiler.compiler.pipeline import compile_pareto
from workflow_compiler.dsl.torchlike import WorkflowModule
from workflow_compiler.workflows.dsl_registry import get_workflow_module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def test_compile_predict_workflow_owned_math(tmp_path: Path):
    full_structure_id = get_workflow_module("math").get_full_structure()["structure_id"]
    detailed_results = {
        "programmer": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.7, "avg_input_tokens": 10, "avg_output_tokens": 20}
            ]
        },
        "refine_solver": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.8, "avg_input_tokens": 12, "avg_output_tokens": 16}
            ]
        },
        "detailed_solver": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.75, "avg_input_tokens": 14, "avg_output_tokens": 18}
            ]
        },
        "generate_solver": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.72, "avg_input_tokens": 11, "avg_output_tokens": 15}
            ]
        },
        "sc_ensemble": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.95, "avg_input_tokens": 8, "avg_output_tokens": 12}
            ]
        },
    }
    trace_training = {
        "training_data": [
            {
                "problem": "Compute 1+1.",
                "original_sample": {"problem": "Compute 1+1.", "unique_id": "m_0001"},
            }
        ]
    }
    latency = {
        "test-llm": [{"prefill_tok_per_s": 1000.0, "decode_tok_per_s": 500.0}],
    }

    detailed_file = tmp_path / "detailed_results.json"
    trace_file = tmp_path / "trace_training_data.json"
    latency_file = tmp_path / "latency.json"
    output_file = tmp_path / "compiled_configs.json"

    _write_json(detailed_file, detailed_results)
    _write_json(trace_file, trace_training)
    _write_json(latency_file, latency)

    compiled = compile_pareto(
        workflow_type="math",
        detailed_results=[str(detailed_file)],
        trace_data=str(trace_file),
        latency_file=str(latency_file),
        output_file=str(output_file),
        search_space={
            "search_axes": ["model", "budget", "structure"],
            "structures": [full_structure_id],
        },
    )

    assert output_file.exists()
    assert compiled["schema_version"] == "flowcompile.compiled.v2"
    assert compiled["workflow_type"] == "math"
    assert "configs" in compiled
    assert "runtime_budget_presets" in compiled
    assert "levels" not in compiled
    assert compiled["configs"], "expected at least one compiled config"
    assert set(compiled["runtime_budget_presets"].keys()) == {"low", "medium", "high", "xhigh"}
    compiled_config_ids = {item["config_id"] for item in compiled["configs"]}
    for preset_cfg in compiled["runtime_budget_presets"].values():
        assert preset_cfg["config_id"] in compiled_config_ids

    cfg = compiled["configs"][0]
    assert cfg["workflow_type"] == "math"
    assert cfg["structure_id"] == full_structure_id
    assert "programmer" in cfg["agents"]
    assert "refine_solver" in cfg["agents"]
    assert "detailed_solver" in cfg["agents"]
    assert "generate_solver" in cfg["agents"]
    assert "sc_ensemble" in cfg["agents"]
    assert "metrics" in cfg
    assert "expected_accuracy" in cfg["metrics"]
    assert "expected_latency" in cfg["metrics"]

    with open(output_file, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert set(persisted["runtime_budget_presets"].keys()) == {"low", "medium", "high", "xhigh"}


def test_compile_predict_saves_subagent_plots_to_figures_dir(monkeypatch, tmp_path: Path):
    full_structure_id = get_workflow_module("math").get_full_structure()["structure_id"]
    detailed_results = {
        "programmer": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.7, "avg_input_tokens": 10, "avg_output_tokens": 20}
            ]
        },
        "refine_solver": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.8, "avg_input_tokens": 12, "avg_output_tokens": 16}
            ]
        },
        "detailed_solver": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.75, "avg_input_tokens": 14, "avg_output_tokens": 18}
            ]
        },
        "generate_solver": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.72, "avg_input_tokens": 11, "avg_output_tokens": 15}
            ]
        },
        "sc_ensemble": {
            "test-llm_budget_200": [
                {"problem": "Compute 1+1.", "accuracy": 0.95, "avg_input_tokens": 8, "avg_output_tokens": 12}
            ]
        },
    }
    trace_training = {
        "training_data": [
            {
                "problem": "Compute 1+1.",
                "original_sample": {"problem": "Compute 1+1.", "unique_id": "m_0001"},
            }
        ]
    }
    latency = {
        "test-llm": [{"prefill_tok_per_s": 1000.0, "decode_tok_per_s": 500.0}],
    }

    detailed_file = tmp_path / "detailed_results.json"
    trace_file = tmp_path / "trace_training_data.json"
    latency_file = tmp_path / "latency.json"
    output_file = tmp_path / "results" / "exp_one" / "02_compile" / "compiled_configs.json"

    _write_json(detailed_file, detailed_results)
    _write_json(trace_file, trace_training)
    _write_json(latency_file, latency)

    captured = {}

    def fake_save_subagent_plots(df_subagents, output_dir, workflow_type):
        captured["subagents"] = set(df_subagents.keys())
        captured["output_dir"] = output_dir
        captured["workflow_type"] = workflow_type
        return {"programmer": str(output_dir / "analyze_programmer_latency_h100.png")}

    monkeypatch.setattr(pipeline, "_save_subagent_latency_score_plots", fake_save_subagent_plots)
    monkeypatch.setattr(pipeline, "_save_latency_score_plot", lambda *_args, **_kwargs: None)

    compiled = pipeline.compile_pareto(
        workflow_type="math",
        detailed_results=[str(detailed_file)],
        trace_data=str(trace_file),
        latency_file=str(latency_file),
        output_file=str(output_file),
        search_space={
            "search_axes": ["model", "budget", "structure"],
            "structures": [full_structure_id],
        },
    )

    expected_figures_dir = output_file.parent / "figures"
    assert captured["workflow_type"] == "math"
    assert captured["output_dir"] == expected_figures_dir
    assert "programmer" in captured["subagents"]

    assert "subagent_score_latency_plots" in compiled["metadata"]
    assert compiled["metadata"]["subagent_score_latency_plots"] == {
        "programmer": str(expected_figures_dir / "analyze_programmer_latency_h100.png")
    }

    with open(output_file, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["metadata"]["subagent_score_latency_plots"] == {
        "programmer": str(expected_figures_dir / "analyze_programmer_latency_h100.png")
    }


def test_extract_plot_model_budget_groups_by_model():
    model, budget = pipeline._extract_plot_model_budget("qwen35-4b_budget_200")
    assert model == "qwen35-4b"
    assert budget == 200.0

    model_local, budget_local = pipeline._extract_plot_model_budget("qwen35-4b-local_budget_1000")
    assert model_local == "qwen35-4b-local"
    assert budget_local == 1000.0

    # Non-standard setting strings fall back to themselves, preserving old behavior safely.
    model_raw, budget_raw = pipeline._extract_plot_model_budget("custom-setting")
    assert model_raw == "custom-setting"
    assert budget_raw == float("inf")


def test_select_runtime_budget_presets_prefers_latency_then_accuracy_extremes():
    configs = [
        {
            "config_id": "cfg_fast",
            "metrics": {"expected_accuracy": 0.2, "expected_latency": 1.0},
        },
        {
            "config_id": "cfg_mid",
            "metrics": {"expected_accuracy": 0.7, "expected_latency": 4.0},
        },
        {
            "config_id": "cfg_acc",
            "metrics": {"expected_accuracy": 0.95, "expected_latency": 10.0},
        },
    ]

    presets = pipeline._select_runtime_budget_presets(configs)
    assert presets["low"]["config_id"] == "cfg_fast"
    assert presets["medium"]["config_id"] == "cfg_mid"
    assert presets["high"]["config_id"] == "cfg_acc"
    assert presets["xhigh"]["config_id"] == "cfg_acc"


def test_extract_runtime_budget_preset_plot_points_merges_duplicate_points():
    points = pipeline._extract_runtime_budget_preset_plot_points(
        {
            "low": {"metrics": {"expected_accuracy": 0.2, "expected_latency": 1.0}},
            "medium": {"metrics": {"expected_accuracy": 0.2, "expected_latency": 1.0}},
            "high": {"metrics": {"expected_accuracy": 0.8, "expected_latency": 5.0}},
        }
    )

    assert len(points) == 2
    point_by_label = {point["label_text"]: point for point in points}
    assert point_by_label["low/medium"]["latency"] == 1.0
    assert point_by_label["low/medium"]["accuracy"] == 0.2
    assert point_by_label["high"]["latency"] == 5.0
    assert point_by_label["high"]["accuracy"] == 0.8


def test_compile_predict_applies_subagent_score_thresholds_before_workflow_generation(monkeypatch, tmp_path: Path):
    class _FakeWorkflow(WorkflowModule):
        workflow_type = "fake"

        def __init__(self):
            super().__init__(name="fake")

        def forward(self, query):
            del query
            raise RuntimeError("unused")

        def infer_agent_names(self):
            return ["agent_a", "agent_b"]

        def normalize_subagent_stats(self, df_subagents):
            return df_subagents

        def compute_configs(self, df_subagents, metadata=None):
            del metadata
            assert set(df_subagents["agent_a"]["setting"].tolist()) == {"a_high"}
            assert set(df_subagents["agent_b"]["setting"].tolist()) == {"b_low", "b_high"}
            return pd.DataFrame(
                [
                    {
                        "workflow_accuracy": 0.8,
                        "workflow_latency": 1.25,
                        "structure_id": "stub",
                        "total_branches": 1,
                        "is_full": True,
                    }
                ]
            )

    monkeypatch.setattr(pipeline, "get_workflow_module", lambda *args, **kwargs: _FakeWorkflow())
    monkeypatch.setattr(pipeline, "convert_to_consolidated", lambda *args, **kwargs: (pd.DataFrame([{"x": 1}]), None))
    monkeypatch.setattr(
        pipeline,
        "build_subagent_stats",
        lambda _df: {
            "agent_a": pd.DataFrame(
                [
                    {"setting": "a_low", "accuracy": 0.2, "latency": 1.0, "input_tokens": 1.0, "output_tokens": 1.0},
                    {"setting": "a_high", "accuracy": 0.9, "latency": 2.0, "input_tokens": 1.0, "output_tokens": 1.0},
                ]
            ),
            "agent_b": pd.DataFrame(
                [
                    {"setting": "b_low", "accuracy": 0.4, "latency": 1.0, "input_tokens": 1.0, "output_tokens": 1.0},
                    {"setting": "b_high", "accuracy": 0.7, "latency": 2.0, "input_tokens": 1.0, "output_tokens": 1.0},
                ]
            ),
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_build_compiled_configs",
        lambda *args, **kwargs: {
            "configs": [
                {
                    "config_id": "cfg_0000",
                    "workflow_type": "fake",
                    "structure_id": "stub",
                    "agents": {},
                    "metrics": {"expected_accuracy": 0.8, "expected_latency": 1.25},
                }
            ]
        },
    )
    monkeypatch.setattr(pipeline, "_save_latency_score_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_save_subagent_latency_score_plots", lambda *args, **kwargs: {})

    output_file = tmp_path / "compiled_configs.json"
    compiled = pipeline.compile_pareto(
        workflow_type="fake",
        detailed_results=["unused.json"],
        trace_data="unused_trace.json",
        latency_file="unused_latency.json",
        output_file=str(output_file),
        subagent_score_thresholds={"agent_a": 0.6},
    )

    metadata = compiled["metadata"]
    assert metadata["subagent_score_thresholds"] == {"agent_a": 0.6}
    assert metadata["subagent_counts_before_threshold"] == {"agent_a": 2, "agent_b": 2}
    assert metadata["subagent_counts_after_threshold"] == {"agent_a": 1, "agent_b": 2}
    assert metadata["subagent_counts_before_prune"] == {"agent_a": 1, "agent_b": 2}
    assert metadata["subagent_counts_after_prune"] == {"agent_a": 1, "agent_b": 2}


def test_compile_predict_subagent_score_thresholds_reject_unknown_subagent(monkeypatch, tmp_path: Path):
    class _FakeWorkflow(WorkflowModule):
        workflow_type = "fake"

        def __init__(self):
            super().__init__(name="fake")

        def forward(self, query):
            del query
            raise RuntimeError("unused")

        def infer_agent_names(self):
            return ["agent_a"]

        def normalize_subagent_stats(self, df_subagents):
            return df_subagents

        def compute_configs(self, df_subagents, metadata=None):
            del df_subagents, metadata
            return pd.DataFrame()

    monkeypatch.setattr(pipeline, "get_workflow_module", lambda *args, **kwargs: _FakeWorkflow())
    monkeypatch.setattr(pipeline, "convert_to_consolidated", lambda *args, **kwargs: (pd.DataFrame([{"x": 1}]), None))
    monkeypatch.setattr(
        pipeline,
        "build_subagent_stats",
        lambda _df: {
            "agent_a": pd.DataFrame(
                [{"setting": "a_high", "accuracy": 0.9, "latency": 2.0, "input_tokens": 1.0, "output_tokens": 1.0}]
            )
        },
    )

    with pytest.raises(ValueError, match="Unknown subagent\\(s\\) in predict_subagent_score_thresholds"):
        pipeline.compile_pareto(
            workflow_type="fake",
            detailed_results=["unused.json"],
            trace_data="unused_trace.json",
            latency_file="unused_latency.json",
            output_file=str(tmp_path / "compiled_configs.json"),
            subagent_score_thresholds={"missing_agent": 0.5},
        )


def test_compile_predict_subagent_score_thresholds_reject_empty_after_filter(monkeypatch, tmp_path: Path):
    class _FakeWorkflow(WorkflowModule):
        workflow_type = "fake"

        def __init__(self):
            super().__init__(name="fake")

        def forward(self, query):
            del query
            raise RuntimeError("unused")

        def infer_agent_names(self):
            return ["agent_a"]

        def normalize_subagent_stats(self, df_subagents):
            return df_subagents

        def compute_configs(self, df_subagents, metadata=None):
            del df_subagents, metadata
            return pd.DataFrame()

    monkeypatch.setattr(pipeline, "get_workflow_module", lambda *args, **kwargs: _FakeWorkflow())
    monkeypatch.setattr(pipeline, "convert_to_consolidated", lambda *args, **kwargs: (pd.DataFrame([{"x": 1}]), None))
    monkeypatch.setattr(
        pipeline,
        "build_subagent_stats",
        lambda _df: {
            "agent_a": pd.DataFrame(
                [{"setting": "a_low", "accuracy": 0.4, "latency": 2.0, "input_tokens": 1.0, "output_tokens": 1.0}]
            )
        },
    )

    with pytest.raises(ValueError, match="removed all settings for subagent 'agent_a'"):
        pipeline.compile_pareto(
            workflow_type="fake",
            detailed_results=["unused.json"],
            trace_data="unused_trace.json",
            latency_file="unused_latency.json",
            output_file=str(tmp_path / "compiled_configs.json"),
            subagent_score_thresholds={"agent_a": 0.5},
        )


def test_compile_predict_persists_workflow_loops_metadata(monkeypatch, tmp_path: Path):
    class _FakeWorkflow(WorkflowModule):
        workflow_type = "fake"

        def __init__(self):
            super().__init__(name="fake")

        def forward(self, query):
            del query
            raise RuntimeError("unused")

        def infer_agent_names(self):
            return ["a"]

        def normalize_subagent_stats(self, df_subagents):
            return df_subagents

        def compute_configs(self, df_subagents, metadata=None):
            del df_subagents
            assert metadata["workflow_loops"] == [
                {
                    "name": "email_loop",
                    "count": 20,
                    "map_nodes": ["summarize_each", "classify"],
                    "reduce_node": "overview",
                }
            ]
            df = pd.DataFrame(
                [
                    {
                        "workflow_accuracy": 0.5,
                        "workflow_latency": 1.25,
                        "structure_id": "stub",
                        "total_branches": 1,
                        "is_full": True,
                    }
                ]
            )
            df.attrs["search_space_resolved"] = {}
            return df

    monkeypatch.setattr(pipeline, "get_workflow_module", lambda *args, **kwargs: _FakeWorkflow())
    monkeypatch.setattr(pipeline, "convert_to_consolidated", lambda *args, **kwargs: (pd.DataFrame([{"x": 1}]), None))
    monkeypatch.setattr(pipeline, "build_subagent_stats", lambda df: {})
    monkeypatch.setattr(
        pipeline,
        "_build_compiled_configs",
        lambda *args, **kwargs: {
            "configs": [
                {
                    "config_id": "cfg_0000",
                    "workflow_type": "fake",
                    "structure_id": "stub",
                    "agents": {},
                    "metrics": {"expected_accuracy": 0.5, "expected_latency": 1.25},
                }
            ]
        },
    )
    monkeypatch.setattr(pipeline, "_save_latency_score_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_save_subagent_latency_score_plots", lambda *args, **kwargs: {})

    output_file = tmp_path / "compiled_configs.json"
    compiled = pipeline.compile_pareto(
        workflow_type="fake",
        detailed_results=["unused.json"],
        trace_data="unused_trace.json",
        latency_file="unused_latency.json",
        output_file=str(output_file),
        workflow_loops=[
            {
                "name": "email_loop",
                "count": 20,
                "map_nodes": ["summarize_each", "classify"],
                "reduce_node": "overview",
            }
        ],
    )

    assert compiled["metadata"]["workflow_loops"] == [
        {
            "name": "email_loop",
            "count": 20,
            "map_nodes": ["summarize_each", "classify"],
            "reduce_node": "overview",
        }
    ]
    with open(output_file, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["metadata"]["workflow_loops"] == [
        {
            "name": "email_loop",
            "count": 20,
            "map_nodes": ["summarize_each", "classify"],
            "reduce_node": "overview",
        }
    ]
