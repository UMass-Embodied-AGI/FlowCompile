from types import SimpleNamespace
import os

from flowcompile.core import cli


def _dummy_prepare_args():
    return SimpleNamespace(
        task=None,
        llm=None,
        experiment_id=None,
        file_path=None,
        debug=False,
        trace_data=None,
        output=None,
        config=None,
        model=None,
        max_samples=None,
        num_workers=None,
        individual=False,
    )


def test_compile_prepare_data_runs_ground_truth_then_agent_dataset(monkeypatch):
    calls = []

    def fake_gt(args, cfg):
        calls.append("gt")
        return 0

    def fake_ad(args, cfg):
        calls.append("ad")
        return 0

    monkeypatch.setattr(cli, "cmd_compile_ground_truth", fake_gt)
    monkeypatch.setattr(cli, "cmd_compile_agent_dataset", fake_ad)

    result = cli.cmd_compile_prepare_data(_dummy_prepare_args(), {})
    assert result == 0
    assert calls == ["gt", "ad"]


def test_compile_latency_defaults_to_openai_backend(monkeypatch):
    captured = {}

    def fake_run_latency_benchmark(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "run_latency_benchmark", fake_run_latency_benchmark)

    args = SimpleNamespace(
        models=None,
        output_json=None,
        prompt_file=None,
        batch_size=None,
        batch_sizes=None,
        max_new_tokens=None,
        dtype=None,
        tp=None,
        gpu_mem_util=None,
        seed=None,
        model_config_path=None,
        backend=None,
    )
    cfg = {
        "schema_version": "flowcompile.flat.v1",
        "experiment_id": "lat_exp",
        "workflow_type": "math",
        "dataset": "MATH500",
        "model_config": "configs/config.yaml",
        "validate_file": "data/math500_validate.jsonl",
        "test_file": "data/math500_test.jsonl",
        "search_axes": ["model", "budget", "structure"],
        "search_budgets": [10, 200],
        "latency_models": ["Qwen/Qwen3-4B"],
    }

    assert cli.cmd_compile_latency(args, cfg) == 0
    assert captured["backend"] == "openai"


def test_compile_agent_dataset_defaults_to_gpt5_filter_model(monkeypatch):
    captured = {}

    def fake_run_agent_dataset(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "run_agent_dataset", fake_run_agent_dataset)

    args = SimpleNamespace(
        trace_data="results/demo/trace.jsonl",
        output="results/demo/aggregated_training_data.json",
        config="configs/config.yaml",
        model=None,
        max_samples=None,
        num_workers=None,
        individual=False,
    )

    assert cli.cmd_compile_agent_dataset(args, {}) == 0
    assert captured["model"] == "gpt-5"


def test_compile_agent_dataset_autodetect_prefers_profile_trace(monkeypatch, tmp_path):
    captured = {}

    def fake_run_agent_dataset(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "run_agent_dataset", fake_run_agent_dataset)
    monkeypatch.chdir(tmp_path)

    exp = "hotpotqa"
    legacy_trace = tmp_path / "results" / exp / "hotpotqa_dsl_agent_20260213_000000" / "trace.jsonl"
    profile_trace = (
        tmp_path
        / "results"
        / exp
        / "01_profile"
        / "hotpotqa_dsl_agent_20260213_013900"
        / "trace.jsonl"
    )
    legacy_trace.parent.mkdir(parents=True, exist_ok=True)
    profile_trace.parent.mkdir(parents=True, exist_ok=True)
    legacy_trace.write_text("{}", encoding="utf-8")
    profile_trace.write_text("{}", encoding="utf-8")

    # Ensure profile trace is newer than the legacy root trace.
    os.utime(legacy_trace, (100, 100))
    os.utime(profile_trace, (200, 200))

    args = SimpleNamespace(
        trace_data=None,
        output=None,
        config=None,
        model=None,
        max_samples=None,
        num_workers=None,
        individual=False,
    )

    assert cli.cmd_compile_agent_dataset(args, {"experiment_id": exp}) == 0
    assert captured["trace_path"] == str(profile_trace.relative_to(tmp_path))
    assert captured["output"] == f"results/{exp}/01_profile/aggregated_training_data.json"
