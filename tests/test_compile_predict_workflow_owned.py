from __future__ import annotations

import json
from pathlib import Path

from workflow_compiler.compiler import pipeline
from workflow_compiler.compiler.pipeline import compile_pareto
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
    assert "levels" not in compiled
    assert compiled["configs"], "expected at least one compiled config"

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
