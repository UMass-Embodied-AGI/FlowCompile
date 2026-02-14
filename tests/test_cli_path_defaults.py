import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflow_compiler.core import cli


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_compile_predict_uses_canonical_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_defaults"
    root = tmp_path / "results" / exp

    _write_json(root / "01_profile" / "benchmark_20260212_000000" / "detailed_results.json", {})
    _write_json(root / "01_profile" / "aggregated_training_data.json", {"training_data": []})
    _write_json(root / "01_profile" / "latency_benchmark.json", {})

    captured = {}

    def fake_compile_pareto(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "compile_pareto", fake_compile_pareto)

    args = SimpleNamespace(
        workflow_type=None,
        detailed_results=None,
        trace_data=None,
        latency_file=None,
        output_file=None,
        plot_file=None,
        include_all=None,
        prune_subagents=None,
        search_axes=None,
        search_models=None,
        search_budgets=None,
        search_structures=None,
        search_agent_models=None,
        search_agent_budgets=None,
    )
    cfg = {
        "compile": {
            "experiment_id": exp,
            "workflow_type": "math",
            "predict": {},
        }
    }

    assert cli.cmd_compile_predict(args, cfg) == 0
    assert captured["trace_data"] == f"results/{exp}/01_profile/aggregated_training_data.json"
    assert captured["latency_file"] == f"results/{exp}/01_profile/latency_benchmark.json"
    assert captured["output_file"] == f"results/{exp}/02_compile/compiled_configs.json"
    assert captured["plot_file"] == f"results/{exp}/02_compile/figures/compiled_latency_vs_score.png"
    assert captured["detailed_results"] == [
        f"results/{exp}/01_profile/benchmark_20260212_000000/detailed_results.json"
    ]


def test_test_defaults_config_and_output_dir(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_validate"
    _write_json(
        tmp_path / "results" / exp / "02_compile" / "compiled_configs.json",
        {"schema_version": "flowcompile.compiled.v2", "configs": []},
    )

    captured = {}

    async def fake_run_validation(ns):
        captured["ns"] = ns
        return 0

    monkeypatch.setattr(cli, "run_validation", fake_run_validation)

    args = SimpleNamespace(
        experiment_id=None,
        config_file=None,
        output_dir=None,
        pareto_sample_n=None,
        parallel=None,
        split=None,
        dataset=None,
        data_path=None,
        random_seed=None,
        start_idx=None,
        end_idx=None,
        max_tasks=None,
    )
    cfg = {"compile": {"experiment_id": exp}, "test": {}}

    assert cli.cmd_test(args, cfg) == 0
    ns = captured["ns"]
    assert ns.config_file == f"results/{exp}/02_compile/compiled_configs.json"
    assert ns.output_dir == f"results/{exp}/03_test"


def test_runtime_knn_uses_experiment_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_knn"
    root = tmp_path / "results" / exp / "01_profile"
    _write_json(root / "benchmark_1" / "detailed_results.json", {})
    _write_json(root / "aggregated_training_data.json", {"training_data": []})
    _write_json(root / "latency_benchmark.json", {})

    captured = {}

    def fake_run_knn(ns):
        captured["ns"] = ns
        return 0

    monkeypatch.setattr(cli, "run_knn", fake_run_knn)

    args = SimpleNamespace(
        experiment_id=None,
        workflow_type=None,
        detailed_results=None,
        trace_data=None,
        latency_file=None,
        test_data=None,
        k=None,
        embedding_model=None,
        max_length=None,
        batch_size=None,
        embedding_cache_file=None,
        accuracy_thresholds=None,
        output_dir=None,
        use_cached_consolidation=False,
        data_files=None,
        search_axes=None,
        search_models=None,
        search_budgets=None,
        search_structures=None,
        search_agent_models=None,
        search_agent_budgets=None,
    )
    cfg = {
        "compile": {"experiment_id": exp, "workflow_type": "math"},
        "runtime": {"workflow_type": "math", "knn": {}},
    }

    assert cli.cmd_runtime_knn(args, cfg) == 0
    ns = captured["ns"]
    assert ns.trace_data == f"results/{exp}/01_profile/aggregated_training_data.json"
    assert ns.latency_file == f"results/{exp}/01_profile/latency_benchmark.json"
    assert ns.output_dir == f"results/{exp}/knn"
    assert ns.test_data == "data/ours/math_test.jsonl"


def test_compile_predict_rejects_noncanonical_latency_path(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_defaults"
    root = tmp_path / "results" / exp

    _write_json(root / "01_profile" / "benchmark_20260212_000000" / "detailed_results.json", {})
    _write_json(root / "01_profile" / "aggregated_training_data.json", {"training_data": []})
    _write_json(root / "01_profile" / "latency_benchmark.json", {})
    _write_json(tmp_path / "custom" / "latency.json", {})

    args = SimpleNamespace(
        workflow_type=None,
        detailed_results=None,
        trace_data=None,
        latency_file="custom/latency.json",
        output_file=None,
        plot_file=None,
        include_all=None,
        prune_subagents=None,
        search_axes=None,
        search_models=None,
        search_budgets=None,
        search_structures=None,
        search_agent_models=None,
        search_agent_budgets=None,
    )
    cfg = {
        "compile": {
            "experiment_id": exp,
            "workflow_type": "math",
            "predict": {},
        }
    }

    with pytest.raises(SystemExit, match="must be the canonical path"):
        cli.cmd_compile_predict(args, cfg)


def test_runtime_knn_rejects_noncanonical_latency_path(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_knn"
    root = tmp_path / "results" / exp / "01_profile"
    _write_json(root / "benchmark_1" / "detailed_results.json", {})
    _write_json(root / "aggregated_training_data.json", {"training_data": []})
    _write_json(root / "latency_benchmark.json", {})
    _write_json(tmp_path / "custom" / "latency.json", {})

    args = SimpleNamespace(
        experiment_id=None,
        workflow_type=None,
        detailed_results=None,
        trace_data=None,
        latency_file="custom/latency.json",
        test_data=None,
        k=None,
        embedding_model=None,
        max_length=None,
        batch_size=None,
        embedding_cache_file=None,
        accuracy_thresholds=None,
        output_dir=None,
        use_cached_consolidation=False,
        data_files=None,
        search_axes=None,
        search_models=None,
        search_budgets=None,
        search_structures=None,
        search_agent_models=None,
        search_agent_budgets=None,
    )
    cfg = {
        "compile": {"experiment_id": exp, "workflow_type": "math"},
        "runtime": {"workflow_type": "math", "knn": {}},
    }

    with pytest.raises(SystemExit, match="must be the canonical path"):
        cli.cmd_runtime_knn(args, cfg)
