import asyncio
from pathlib import Path

import pytest

from workflow_compiler.compiler.profiling import get_experiment_config
from workflow_compiler.compiler import profiling


def test_get_experiment_config_prefers_01_profile_aggregated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    exp = "hotpotqa"
    profile_agg = tmp_path / "results" / exp / "01_profile" / "aggregated_training_data.json"
    profile_agg.parent.mkdir(parents=True, exist_ok=True)
    profile_agg.write_text("{}", encoding="utf-8")

    cfg = get_experiment_config(exp)

    assert cfg["training_data_path"] == f"results/{exp}/01_profile/aggregated_training_data.json"
    assert cfg["output_dir"] == f"results/{exp}/01_profile"


def test_get_experiment_config_falls_back_to_legacy_data_aggregated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    exp = "hotpotqa"
    legacy_agg = tmp_path / "results" / exp / "data" / "aggregated_training_data.json"
    legacy_agg.parent.mkdir(parents=True, exist_ok=True)
    legacy_agg.write_text("{}", encoding="utf-8")

    cfg = get_experiment_config(exp)

    assert cfg["training_data_path"] == f"results/{exp}/data/aggregated_training_data.json"
    assert cfg["output_dir"] == f"results/{exp}/01_profile"


def test_run_profiling_closes_runner_on_success(monkeypatch):
    state = {"closed": False, "saved": False, "ran": False}

    class FakeRunner:
        async def run_benchmark(self):
            state["ran"] = True

        def save_results(self):
            state["saved"] = True
            return Path("results/exp/01_profile/benchmark_00000000_000000")

        async def aclose(self):
            state["closed"] = True

    monkeypatch.setattr(
        profiling.BenchmarkConfig,
        "initialize_from_experiment_id",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(profiling, "BenchmarkRunner", FakeRunner)

    output = asyncio.run(profiling.run_profiling(experiment_id="exp", max_concurrent=1))

    assert output == Path("results/exp/01_profile/benchmark_00000000_000000")
    assert state == {"closed": True, "saved": True, "ran": True}


def test_run_profiling_closes_runner_on_failure(monkeypatch):
    state = {"closed": False, "ran": False}

    class FakeRunner:
        async def run_benchmark(self):
            state["ran"] = True
            raise RuntimeError("boom")

        def save_results(self):  # pragma: no cover
            raise AssertionError("save_results should not be called when run_benchmark fails")

        async def aclose(self):
            state["closed"] = True

    monkeypatch.setattr(
        profiling.BenchmarkConfig,
        "initialize_from_experiment_id",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(profiling, "BenchmarkRunner", FakeRunner)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(profiling.run_profiling(experiment_id="exp", max_concurrent=1))

    assert state == {"closed": True, "ran": True}


def test_initialize_from_experiment_id_warns_when_builtin_judge_policies_are_ignored(monkeypatch, capsys):
    monkeypatch.setattr(
        profiling,
        "get_experiment_config",
        lambda *args, **kwargs: {
            "training_data_path": "results/exp/01_profile/aggregated_training_data.json",
            "output_dir": "results/exp/01_profile",
            "workflow_type": "math",
            "search_budgets": [10, 20],
            "agent_names": ["programmer"],
            "workflow_module": object(),
            "workflow_judges": {"programmer": object()},
            "openclaw_lobster_workflow_file": None,
            "openclaw_agent_policies": {},
        },
    )

    profiling.BenchmarkConfig.initialize_from_experiment_id(
        "exp",
        judge_policies={"programmer": {"mode": "semantic_llm", "prompt": "unused"}},
    )

    assert "judge_policies are ignored for built-in workflows" in capsys.readouterr().out
