import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflow_compiler.core import cli
from workflow_compiler.runtime.selector import RUNTIME_PREFERENCE_BUDGET_PRESETS


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _runtime_args(**overrides):
    args = dict(
        query=None,
        query_id=None,
        compiled=None,
        queries=None,
        output_dir=None,
        workflow_type="math",
        strategy="preference",
        budget="0.5",
        min_accuracy=None,
        max_latency=None,
        knn_k=20,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


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


def test_runtime_infer_batch_uses_experiment_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_infer"
    compiled_path = tmp_path / "results" / exp / "02_compile" / "compiled_configs.json"
    _write_json(compiled_path, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    queries_path = tmp_path / "queries.jsonl"
    _write_jsonl(queries_path, [{"id": "q1", "problem": "Solve 1+1"}])

    captured = {}

    def fake_infer_runtime_batch(**kwargs):
        captured.update(kwargs)
        return [{"query_id": "q1", "answer": "2"}]

    monkeypatch.setattr(cli, "infer_runtime_batch", fake_infer_runtime_batch)

    args = _runtime_args(
        queries=str(queries_path),
        workflow_type=None,
    )
    cfg = {
        "compile": {"experiment_id": exp, "workflow_type": "math"},
    }

    assert cli.cmd_runtime_infer(args, cfg) == 0
    assert captured["workflow_type"] == "math"
    assert len(captured["configs"]) == 1
    out_file = tmp_path / "results" / exp / "runtime" / "outputs" / "runtime_results.jsonl"
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["answer"] == "2"


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


def test_runtime_infer_single_query_prints_human_readable_summary(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.chdir(tmp_path)
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})

    def fake_infer_runtime(**kwargs):
        assert kwargs["query"] == "Solve 1+1"
        assert kwargs["query_id"] == "q1"
        return {
            "query": {"id": "q1", "problem": "Solve 1+1"},
            "selected_config": {
                "config_id": "cfg_0000",
                "structure_id": "full",
                "agents": {
                    "generate_solver": {
                        "setting": "qwen3-1.7b_budget_10",
                        "model": "qwen3-1.7b",
                        "budget": 10,
                    },
                    "sc_ensemble": {
                        "setting": "qwen3-8b_budget_10",
                        "model": "qwen3-8b",
                        "budget": 10,
                    },
                },
            },
            "answer": "2",
            "workflow_output": "2",
            "routing_runtime_seconds": 0.4567,
            "actual_runtime_seconds": 4.2374,
            "query_id": "q1",
            "config_id": "cfg_0000",
            "structure_id": "full",
            "output_dir": "runtime_outputs/q1",
        }

    monkeypatch.setattr(cli, "infer_runtime", fake_infer_runtime)

    args = _runtime_args(
        query="Solve 1+1",
        query_id="q1",
        compiled=str(compiled_file),
    )
    cfg = {}

    assert cli.cmd_runtime_infer(args, cfg) == 0
    printed = capsys.readouterr().out.strip()
    assert "Used Config" in printed
    assert "Config ID: cfg_0000" in printed
    assert "Structure ID: full" in printed
    assert "Sub-agents:" in printed
    assert "generate_solver: setting=qwen3-1.7b_budget_10, model=qwen3-1.7b, budget=10" in printed
    assert "sc_ensemble: setting=qwen3-8b_budget_10, model=qwen3-8b, budget=10" in printed
    assert "Workflow Output" in printed
    assert "\n  2\n" in f"\n{printed}\n"
    assert "Routing Runtime" in printed
    assert "0.457s" in printed
    assert "Actual Runtime" in printed
    assert "4.237s" in printed
    assert "Metadata" in printed
    assert "Query ID: q1" in printed
    assert "Output Dir: runtime_outputs/q1" in printed


def test_runtime_infer_batch_prints_brief_summary(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.chdir(tmp_path)
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    queries_file = tmp_path / "queries.jsonl"
    _write_jsonl(queries_file, [{"id": "q1", "problem": "Solve 1+1"}])

    def fake_infer_runtime_batch(**kwargs):
        return [
            {
                "query": {"id": "q1", "problem": "Solve 1+1"},
                "selected_config": {"config_id": "cfg_0000"},
                "answer": "2",
                "routing_runtime_seconds": None,
                "query_id": "q1",
                "config_id": "cfg_0000",
                "structure_id": "full",
                "output_dir": "runtime_outputs/q1",
            }
        ]

    monkeypatch.setattr(cli, "infer_runtime_batch", fake_infer_runtime_batch)

    args = _runtime_args(
        queries=str(queries_file),
        compiled=str(compiled_file),
    )

    assert cli.cmd_runtime_infer(args, {}) == 0
    printed = capsys.readouterr().out.strip()
    assert "Runtime Infer" in printed
    assert "Queries processed: 1" in printed
    assert "runtime_results.jsonl" in printed
    out_file = tmp_path / "runtime_outputs" / "runtime_results.jsonl"
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["answer"] == "2"
    assert payload["routing_runtime_seconds"] is None
    assert "workflow_output" not in payload
    assert "actual_runtime_seconds" not in payload


def test_runtime_infer_rejects_query_and_queries_together(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    queries_file = tmp_path / "queries.jsonl"
    _write_jsonl(queries_file, [{"id": "q1", "problem": "Solve 1+1"}])
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        queries=str(queries_file),
    )
    with pytest.raises(SystemExit, match="Provide exactly one of --query or --queries"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_requires_query_or_queries_from_cli(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        compiled=str(compiled_file),
    )
    with pytest.raises(SystemExit, match="required via CLI"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_requires_strategy_from_cli(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy=None,
    )
    with pytest.raises(SystemExit, match="--strategy is required via CLI"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_constraint_requires_at_least_one_constraint(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="constraint",
        budget=None,
        min_accuracy=None,
        max_latency=None,
    )
    with pytest.raises(SystemExit, match="requires at least one of --min-accuracy or --max-latency"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_constraint_rejects_budget(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="constraint",
        budget="0.5",
        min_accuracy=0.8,
    )
    with pytest.raises(SystemExit, match="--budget is only valid with --strategy preference"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_preference_requires_budget(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="preference",
        budget=None,
    )
    with pytest.raises(SystemExit, match="--strategy preference requires --budget"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_preference_rejects_constraint_flags(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="preference",
        budget="0.5",
        max_latency=2.0,
    )
    with pytest.raises(SystemExit, match="--min-accuracy/--max-latency are only valid with --strategy constraint"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_rejects_deprecated_flat_runtime_routing_keys_in_yaml(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="preference",
        budget="0.5",
    )
    cfg = {
        "runtime_strategy": "preference",
    }
    with pytest.raises(SystemExit, match="Remove YAML key\\(s\\): runtime_strategy"):
        cli.cmd_runtime_infer(args, cfg)


def test_runtime_infer_rejects_deprecated_nested_runtime_routing_keys_in_yaml(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="preference",
        budget="0.5",
    )
    cfg = {
        "runtime": {"alpha": 0.2},
    }
    with pytest.raises(SystemExit, match="Remove YAML key\\(s\\): runtime.alpha"):
        cli.cmd_runtime_infer(args, cfg)


def test_runtime_infer_rejects_deprecated_flat_runtime_budget_key_in_yaml(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="preference",
        budget="0.5",
    )
    cfg = {
        "runtime_budget": "high",
    }
    with pytest.raises(SystemExit, match="Remove YAML key\\(s\\): runtime_budget"):
        cli.cmd_runtime_infer(args, cfg)


def test_runtime_infer_rejects_deprecated_nested_runtime_budget_key_in_yaml(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="preference",
        budget="0.5",
    )
    cfg = {
        "runtime": {"budget": "high"},
    }
    with pytest.raises(SystemExit, match="Remove YAML key\\(s\\): runtime.budget"):
        cli.cmd_runtime_infer(args, cfg)


@pytest.mark.parametrize(
    ("raw_budget", "expected_budget"),
    [
        ("0.2", 0.2),
        ("low", RUNTIME_PREFERENCE_BUDGET_PRESETS["low"]),
        ("medium", RUNTIME_PREFERENCE_BUDGET_PRESETS["medium"]),
        ("high", RUNTIME_PREFERENCE_BUDGET_PRESETS["high"]),
        ("xhigh", RUNTIME_PREFERENCE_BUDGET_PRESETS["xhigh"]),
    ],
)
def test_runtime_infer_preference_parses_budget_values(monkeypatch, tmp_path: Path, raw_budget, expected_budget):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    captured = {}

    def fake_infer_runtime(**kwargs):
        captured.update(kwargs)
        return {
            "query": {"id": "q1", "problem": "Solve 1+1"},
            "selected_config": {"config_id": "cfg_0000", "structure_id": "full", "agents": {}},
            "answer": "2",
            "workflow_output": "2",
            "actual_runtime_seconds": 1.0,
            "query_id": "q1",
            "config_id": "cfg_0000",
            "structure_id": "full",
            "output_dir": "runtime_outputs/q1",
        }

    monkeypatch.setattr(cli, "infer_runtime", fake_infer_runtime)

    args = _runtime_args(
        query="Solve 1+1",
        query_id="q1",
        compiled=str(compiled_file),
        strategy="preference",
        budget=raw_budget,
    )

    assert cli.cmd_runtime_infer(args, {}) == 0
    assert captured["budget"] == pytest.approx(expected_budget)


def test_runtime_infer_rejects_invalid_budget_value(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="preference",
        budget="ultra",
    )
    with pytest.raises(SystemExit, match="--budget must be one of low, medium, high, xhigh, or a float between 0.0 and 1.0"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_rejects_out_of_range_budget_value(tmp_path: Path):
    compiled_file = tmp_path / "compiled_configs.json"
    _write_json(compiled_file, {"schema_version": "flowcompile.compiled.v2", "configs": [{"config_id": "cfg_0000"}]})
    args = _runtime_args(
        query="Solve 1+1",
        compiled=str(compiled_file),
        strategy="preference",
        budget="1.5",
    )
    with pytest.raises(SystemExit, match="--budget must be between 0.0 and 1.0"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_knn_router_requires_budget(tmp_path: Path):
    args = _runtime_args(
        query="Solve 1+1",
        compiled=None,
        strategy="knn-router",
        budget=None,
    )
    with pytest.raises(SystemExit, match="--strategy knn-router requires --budget"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_knn_router_rejects_constraint_flags(tmp_path: Path):
    args = _runtime_args(
        query="Solve 1+1",
        compiled=None,
        strategy="knn-router",
        budget="0.5",
        max_latency=2.0,
    )
    with pytest.raises(SystemExit, match="not valid with --strategy knn-router"):
        cli.cmd_runtime_infer(args, {})


def test_runtime_infer_knn_router_builds_router_from_experiment_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_knn"
    root = tmp_path / "results" / exp
    compiled_path = root / "02_compile" / "compiled_configs.json"
    _write_json(
        compiled_path,
        {
            "schema_version": "flowcompile.compiled.v2",
            "configs": [],
            "metadata": {"search_space": {"search_axes": ["budget", "model", "structure"], "budgets": [10]}},
        },
    )
    _write_json(root / "01_profile" / "benchmark_20260212_000000" / "detailed_results.json", {})
    _write_json(root / "01_profile" / "aggregated_training_data.json", {"training_data": []})
    _write_json(root / "01_profile" / "latency_benchmark.json", {})
    _write_jsonl(tmp_path / "data.jsonl", [{"problem": "Solve 1+1", "unique_id": "q1"}])
    _write_text(
        tmp_path / "configs" / "config.yaml",
        "models:\n  qwen3-4b:\n    model: qwen3-4b\n    hf_model_name: Qwen/Qwen3-4B\n",
    )

    captured = {}

    def fake_consolidate_validation_data(**kwargs):
        captured["consolidate"] = kwargs
        return {"q1": {"query_text": "Solve 1+1", "agents": {}}}

    class FakeRouter:
        def fit_from_query_table(self, query_data_table):
            captured["fit_from_query_table"] = query_data_table

    def fake_get_router(name, **kwargs):
        captured["router_name"] = name
        captured["router_kwargs"] = kwargs
        return FakeRouter()

    def fake_infer_runtime(**kwargs):
        captured["infer_runtime"] = kwargs
        return {
            "query": {"id": "q1", "problem": "Solve 1+1"},
            "selected_config": {"config_id": "knn_cfg_0000", "structure_id": "full", "agents": {}},
            "answer": "2",
            "workflow_output": "2",
            "actual_runtime_seconds": 1.0,
            "query_id": "q1",
            "config_id": "knn_cfg_0000",
            "structure_id": "full",
            "output_dir": "runtime_outputs/q1",
        }

    monkeypatch.setattr(cli, "consolidate_validation_data", fake_consolidate_validation_data)
    monkeypatch.setattr(cli, "get_router", fake_get_router)
    monkeypatch.setattr(cli, "infer_runtime", fake_infer_runtime)

    args = _runtime_args(
        query="Solve 1+1",
        query_id="q1",
        workflow_type=None,
        strategy="knn-router",
        budget="0.5",
        compiled=str(compiled_path),
        knn_k=20,
    )
    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": exp,
        "workflow_type": "math",
        "dataset": "MATH500",
        "model_config": "configs/config.yaml",
        "validate_file": "data.jsonl",
        "test_file": "test.jsonl",
        "search_axes": ["model", "budget", "structure"],
        "search_budgets": [10],
        "latency_models": ["Qwen/Qwen3-4B"],
    }

    assert cli.cmd_runtime_infer(args, cfg) == 0
    assert captured["router_name"] == "knn"
    assert captured["router_kwargs"]["k"] == 20
    assert captured["router_kwargs"]["embedding_model"] == "allenai/longformer-base-4096"
    assert captured["router_kwargs"]["embedding_cache_file"] == f"results/{exp}/01_profile/knn_longformer_embeddings.pkl"
    assert captured["consolidate"]["detailed_results_files"] == [
        f"results/{exp}/01_profile/benchmark_20260212_000000/detailed_results.json"
    ]
    assert captured["consolidate"]["trace_data_file"] == f"results/{exp}/01_profile/aggregated_training_data.json"
    assert captured["consolidate"]["latency_file"] == f"results/{exp}/01_profile/latency_benchmark.json"
    assert captured["consolidate"]["data_files"] == "data.jsonl"
    assert captured["consolidate"]["model_config_path"] == "configs/config.yaml"
    assert captured["infer_runtime"]["router"] is not None


def test_runtime_infer_knn_router_batch_reuses_single_router(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    exp = "exp_knn_batch"
    root = tmp_path / "results" / exp
    compiled_path = root / "02_compile" / "compiled_configs.json"
    _write_json(compiled_path, {"schema_version": "flowcompile.compiled.v2", "configs": []})
    _write_json(root / "01_profile" / "benchmark_20260212_000000" / "detailed_results.json", {})
    _write_json(root / "01_profile" / "aggregated_training_data.json", {"training_data": []})
    _write_json(root / "01_profile" / "latency_benchmark.json", {})
    _write_jsonl(tmp_path / "data.jsonl", [{"problem": "Solve 1+1", "unique_id": "q1"}])
    queries_path = tmp_path / "queries.jsonl"
    _write_jsonl(queries_path, [{"id": "q1", "problem": "Solve 1+1"}])
    _write_text(
        tmp_path / "configs" / "config.yaml",
        "models:\n  qwen3-4b:\n    model: qwen3-4b\n    hf_model_name: Qwen/Qwen3-4B\n",
    )

    captured = {"routers": []}

    def fake_consolidate_validation_data(**kwargs):
        return {"q1": {"query_text": "Solve 1+1", "agents": {}}}

    class FakeRouter:
        def fit_from_query_table(self, query_data_table):
            captured["fit_from_query_table"] = query_data_table

    def fake_get_router(name, **kwargs):
        router = FakeRouter()
        captured["routers"].append(router)
        return router

    def fake_infer_runtime_batch(**kwargs):
        captured["infer_runtime_batch"] = kwargs
        return [{"query_id": "q1", "answer": "2"}]

    monkeypatch.setattr(cli, "consolidate_validation_data", fake_consolidate_validation_data)
    monkeypatch.setattr(cli, "get_router", fake_get_router)
    monkeypatch.setattr(cli, "infer_runtime_batch", fake_infer_runtime_batch)

    args = _runtime_args(
        queries=str(queries_path),
        workflow_type=None,
        strategy="knn-router",
        budget="0.5",
        compiled=str(compiled_path),
    )
    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": exp,
        "workflow_type": "math",
        "dataset": "MATH500",
        "model_config": "configs/config.yaml",
        "validate_file": "data.jsonl",
        "test_file": "test.jsonl",
        "search_axes": ["model", "budget", "structure"],
        "search_budgets": [10],
        "latency_models": ["Qwen/Qwen3-4B"],
    }

    assert cli.cmd_runtime_infer(args, cfg) == 0
    assert len(captured["routers"]) == 1
    assert captured["infer_runtime_batch"]["router"] is captured["routers"][0]
