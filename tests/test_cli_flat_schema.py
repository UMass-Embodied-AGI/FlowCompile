import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flowcompile.core import cli


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _flat_cfg(**overrides):
    base = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": "exp_flat",
        "workflow_type": "math",
        "dataset": "MATH500",
        "model_config": "configs/config.yaml",
        "validate_file": "data/math500_validate.jsonl",
        "test_file": "data/math500_test.jsonl",
        "search_axes": ["model", "budget", "structure"],
        "search_budgets": [10, 200],
        "search_models": ["qwen3-4b"],
    }
    base.update(overrides)
    return base


def _empty_test_args():
    return SimpleNamespace(
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


def test_load_yaml_rejects_removed_test_keys(tmp_path: Path):
    cfg_path = tmp_path / "removed_test_key.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'schema_version: "flowcompile.flat.v1"',
                'experiment_id: "exp1"',
                'workflow_type: "math"',
                'dataset: "MATH500"',
                'model_config: "configs/config.yaml"',
                "latency_models: ['Qwen/Qwen3-4B']",
                'validate_file: "data/math500_validate.jsonl"',
                'test_file: "data/math500_test.jsonl"',
                "search_axes: ['model', 'budget', 'structure']",
                "search_budgets: [10, 200]",
                "test_limit: 5",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Removed test config key"):
        cli._load_yaml(str(cfg_path))


def _empty_predict_args():
    return SimpleNamespace(
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


def _empty_profile_args():
    return SimpleNamespace(
        experiment_id=None,
        models=None,
        max_samples=None,
        max_concurrent=None,
        debug=False,
        min_samples_per_agent=None,
        search_budgets=None,
    )


def test_load_yaml_rejects_missing_required_key(tmp_path: Path):
    cfg_path = tmp_path / "flat.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'schema_version: "flowcompile.flat.v1"',
                'experiment_id: "exp1"',
                'workflow_type: "math"',
                'dataset: "MATH500"',
                'model_config: "configs/config.yaml"',
                "latency_models: ['Qwen/Qwen3-4B']",
                'validate_file: "data/math500_validate.jsonl"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Missing required flat config key"):
        cli._load_yaml(str(cfg_path))


def test_load_yaml_rejects_legacy_nested_keys(tmp_path: Path):
    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'schema_version: "flowcompile.flat.v1"',
                'experiment_id: "exp1"',
                'workflow_type: "math"',
                'dataset: "MATH500"',
                'model_config: "configs/config.yaml"',
                "latency_models: ['Qwen/Qwen3-4B']",
                'validate_file: "data/math500_validate.jsonl"',
                'test_file: "data/math500_test.jsonl"',
                "search_axes: ['model', 'budget', 'structure']",
                "search_budgets: [10, 200]",
                "compile: {}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Unsupported nested/legacy top-level keys"):
        cli._load_yaml(str(cfg_path))


def test_load_yaml_rejects_validate_process_keys(tmp_path: Path):
    cfg_path = tmp_path / "legacy_validate_keys.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'schema_version: "flowcompile.flat.v1"',
                'experiment_id: "exp1"',
                'workflow_type: "math"',
                'dataset: "MATH500"',
                'model_config: "configs/config.yaml"',
                'validate_file: "data/math500_validate.jsonl"',
                'test_file: "data/math500_test.jsonl"',
                "search_axes: ['model', 'budget', 'structure']",
                "search_budgets: [10, 200]",
                "validate_parallel: 4",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Unsupported keys for test process"):
        cli._load_yaml(str(cfg_path))


def test_test_command_uses_validate_file_and_test_file_by_split(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    async def fake_run_validation(ns):
        captured["ns"] = ns
        return 0

    monkeypatch.setattr(cli, "run_validation", fake_run_validation)
    _write_json(
        tmp_path / "results" / "exp_flat" / "02_compile" / "compiled_configs.json",
        {"schema_version": "flowcompile.compiled.v2", "configs": []},
    )

    cfg_validate = _flat_cfg(test_split="validate")
    assert cli.cmd_test(_empty_test_args(), cfg_validate) == 0
    assert captured["ns"].data_path == "data/math500_validate.jsonl"

    cfg_test = _flat_cfg(test_split="test")
    assert cli.cmd_test(_empty_test_args(), cfg_test) == 0
    assert captured["ns"].data_path == "data/math500_test.jsonl"


def test_test_command_forwards_pareto_sample_n(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    async def fake_run_validation(ns):
        captured["ns"] = ns
        return 0

    monkeypatch.setattr(cli, "run_validation", fake_run_validation)
    _write_json(
        tmp_path / "results" / "exp_flat" / "02_compile" / "compiled_configs.json",
        {"schema_version": "flowcompile.compiled.v2", "configs": []},
    )

    cfg = _flat_cfg(test_pareto_sample_n=7)
    assert cli.cmd_test(_empty_test_args(), cfg) == 0
    assert captured["ns"].pareto_sample_n == 7


def test_test_command_allows_disable_pareto_sampling_with_minus_one(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    async def fake_run_validation(ns):
        captured["ns"] = ns
        return 0

    monkeypatch.setattr(cli, "run_validation", fake_run_validation)
    _write_json(
        tmp_path / "results" / "exp_flat" / "02_compile" / "compiled_configs.json",
        {"schema_version": "flowcompile.compiled.v2", "configs": []},
    )

    cfg = _flat_cfg(test_pareto_sample_n=-1)
    assert cli.cmd_test(_empty_test_args(), cfg) == 0
    assert captured["ns"].pareto_sample_n == -1


def test_test_command_rejects_invalid_pareto_sample_n(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _write_json(
        tmp_path / "results" / "exp_flat" / "02_compile" / "compiled_configs.json",
        {"schema_version": "flowcompile.compiled.v2", "configs": []},
    )
    with pytest.raises(SystemExit, match="--pareto-sample-n must be >= 1, or -1 to disable sampling"):
        cli.cmd_test(_empty_test_args(), _flat_cfg(test_pareto_sample_n=0))

    with pytest.raises(SystemExit, match="--pareto-sample-n must be >= 1, or -1 to disable sampling"):
        cli.cmd_test(_empty_test_args(), _flat_cfg(test_pareto_sample_n=-2))


def test_compile_predict_derives_search_models_from_latency_models(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_flat"
    root = tmp_path / "results" / exp / "01_profile"
    _write_json(root / "benchmark_1" / "detailed_results.json", {})
    _write_json(root / "aggregated_training_data.json", {"training_data": []})
    _write_json(root / "latency_benchmark.json", {})

    model_cfg = tmp_path / "configs" / "config.yaml"
    _write_text(
        model_cfg,
        """
models:
  qwen3-4b:
    api_type: "openai"
    hf_model_name: "Qwen/Qwen3-4B"
""".strip(),
    )

    captured = {}

    def fake_compile_pareto(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "compile_pareto", fake_compile_pareto)

    cfg = _flat_cfg(
        experiment_id=exp,
        model_config=str(model_cfg),
        latency_models=["Qwen/Qwen3-4B"],
    )
    assert cli.cmd_compile_predict(_empty_predict_args(), cfg) == 0
    assert captured["search_space"]["models"] == ["qwen3-4b"]


def test_derive_search_models_errors_for_missing_and_ambiguous_mapping(tmp_path: Path):
    missing_cfg = _flat_cfg(model_config=str(tmp_path / "missing_map.yaml"), latency_models=["Qwen/Qwen3-8B"])
    _write_text(
        Path(missing_cfg["model_config"]),
        """
models:
  qwen3-4b:
    hf_model_name: "Qwen/Qwen3-4B"
""".strip(),
    )
    with pytest.raises(SystemExit, match="Unable to derive search model alias"):
        cli._derive_search_models_from_latency_models(missing_cfg)

    ambiguous_cfg = _flat_cfg(model_config=str(tmp_path / "ambiguous_map.yaml"), latency_models=["Qwen/Qwen3-4B"])
    _write_text(
        Path(ambiguous_cfg["model_config"]),
        """
models:
  qwen3-4b:
    hf_model_name: "Qwen/Qwen3-4B"
  gpt-5-mini:
    model: "qwen3-0.6b"
    hf_model_name: "Qwen/Qwen3-4B"
""".strip(),
    )
    with pytest.raises(SystemExit, match="Ambiguous alias mapping"):
        cli._derive_search_models_from_latency_models(ambiguous_cfg)


def test_load_yaml_sets_latency_models_and_ground_truth_llm_defaults(tmp_path: Path):
    cfg_path = tmp_path / "defaults.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'schema_version: "flowcompile.flat.v1"',
                'experiment_id: "exp1"',
                'workflow_type: "math"',
                'dataset: "MATH500"',
                'model_config: "configs/config.yaml"',
                'validate_file: "data/math500_validate.jsonl"',
                'test_file: "data/math500_test.jsonl"',
                "search_axes: ['model', 'budget', 'structure']",
                "search_budgets: [10, 200]",
            ]
        ),
        encoding="utf-8",
    )

    loaded = cli._load_yaml(str(cfg_path))
    assert loaded["ground_truth_llm"] == "gpt-5-mini"
    assert loaded["ground_truth_task"] == "math500"
    assert loaded["latency_models"] == [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
    ]


def test_compile_ground_truth_defaults_task_and_file_from_dataset_and_validate_file(monkeypatch):
    captured = {}

    async def fake_run_ground_truth(ns):
        captured["ns"] = ns
        return 0

    monkeypatch.setattr(cli, "run_ground_truth", fake_run_ground_truth)
    args = SimpleNamespace(
        task=None,
        llm=None,
        meta_llm=None,
        solver_llm=None,
        programmer_llm=None,
        refine_solver_llm=None,
        detailed_solver_llm=None,
        generate_solver_llm=None,
        answer_generate_llm=None,
        sc_ensemble_llm=None,
        format_answer_llm=None,
        code_generate_llm=None,
        test_llm=None,
        reflection_test_llm=None,
        rewriter_llm=None,
        reader_llm=None,
        answer_reviewer_llm=None,
        mcp_url=None,
        experiment_id=None,
        file_path=None,
        debug=False,
    )
    cfg = _flat_cfg(dataset="HotpotQA", validate_file="data/hotpotqa_validate.jsonl")
    assert cli.cmd_compile_ground_truth(args, cfg) == 0
    ns = captured["ns"]
    assert ns.task == "hotpotqa"
    assert ns.file_path == "data/hotpotqa_validate.jsonl"


def test_cmd_compile_all_runs_latency_prepare_profile_predict_test_in_order(monkeypatch):
    call_order = []

    def _record(name):
        def _inner(_args, _cfg):
            call_order.append(name)
            return 0
        return _inner

    monkeypatch.setattr(cli, "cmd_compile_latency", _record("get-latency"))
    monkeypatch.setattr(cli, "cmd_compile_prepare_data", _record("prepare-data"))
    monkeypatch.setattr(cli, "cmd_compile_profile", _record("profile"))
    monkeypatch.setattr(cli, "cmd_compile_predict", _record("predict"))
    monkeypatch.setattr(cli, "cmd_test", _record("test"))

    assert cli.cmd_compile_all(SimpleNamespace(), {}) == 0
    assert call_order == ["get-latency", "prepare-data", "profile", "predict", "test"]


def test_cmd_compile_profile_uses_flat_min_samples_per_agent(monkeypatch):
    captured = {}

    async def fake_run_profiling(**kwargs):
        captured.update(kwargs)
        return Path("results/exp_flat/01_profile/benchmark_00000000_000000")

    monkeypatch.setattr(cli, "run_profiling", fake_run_profiling)

    cfg = _flat_cfg(min_samples_per_agent=321)
    assert cli.cmd_compile_profile(_empty_profile_args(), cfg) == 0
    assert captured["min_samples_per_agent"] == 321


def test_cmd_compile_profile_prefers_profile_specific_min_samples(monkeypatch):
    captured = {}

    async def fake_run_profiling(**kwargs):
        captured.update(kwargs)
        return Path("results/exp_flat/01_profile/benchmark_00000000_000000")

    monkeypatch.setattr(cli, "run_profiling", fake_run_profiling)

    cfg = _flat_cfg(min_samples_per_agent=321, profile_min_samples_per_agent=123)
    assert cli.cmd_compile_profile(_empty_profile_args(), cfg) == 0
    assert captured["min_samples_per_agent"] == 123
