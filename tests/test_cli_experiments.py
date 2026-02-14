from types import SimpleNamespace

import pytest

from workflow_compiler.core import cli


def test_experiments_correlation_from_config_minimal(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    exp = "hotpotqa"
    workflow_dir = tmp_path / "results" / exp / "03_test"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    latency_file = tmp_path / "results" / exp / "01_profile" / "latency_benchmark.json"
    latency_file.parent.mkdir(parents=True, exist_ok=True)
    latency_file.write_text("{}", encoding="utf-8")

    captured = {}

    def fake_run_correlation(cmd_args):
        captured["cmd_args"] = cmd_args
        return 0

    monkeypatch.setattr(cli, "_run_correlation_experiment", fake_run_correlation)

    args = SimpleNamespace(name="correlation", extra=[])
    cfg = {"experiment_id": exp, "workflow_type": "hotpotqa"}

    assert cli.cmd_experiments(args, cfg) == 0
    assert captured["cmd_args"] == [
        "--workflow-all-results-dir",
        f"results/{exp}/03_test",
        "--workflow-type",
        "hotpotqa",
        "--latency-file",
        f"results/{exp}/01_profile/latency_benchmark.json",
        "--output-dir",
        f"results/{exp}/04_experiments/correlation",
    ]


@pytest.mark.parametrize(
    "name",
    ["correlation"],
)
def test_experiments_requires_config(name):
    args = SimpleNamespace(name=name, extra=[])
    with pytest.raises(SystemExit, match="requires --config"):
        cli.cmd_experiments(args, {})


@pytest.mark.parametrize(
    "name",
    [
        "unified-eval",
        "ablation",
        "average-alpha",
        "pareto-plot",
        "analysis-pipeline",
    ],
)
def test_experiments_rejects_removed_experiment_names(name):
    args = SimpleNamespace(name=name, extra=[])
    cfg = {"experiment_id": "hotpotqa", "workflow_type": "hotpotqa"}

    with pytest.raises(SystemExit, match=f"Unknown experiment script '{name}'"):
        cli.cmd_experiments(args, cfg)


def test_experiments_passthrough_auto_adds_output_dir(monkeypatch):
    captured = {}

    def fake_run_correlation(cmd_args):
        captured["cmd_args"] = cmd_args
        return 0

    monkeypatch.setattr(cli, "_run_correlation_experiment", fake_run_correlation)

    extra = ["--dummy-flag", "x"]
    args = SimpleNamespace(name="correlation", extra=extra)
    cfg = {"experiment_id": "hotpotqa", "workflow_type": "hotpotqa"}

    assert cli.cmd_experiments(args, cfg) == 0
    assert captured["cmd_args"] == [
        "--dummy-flag",
        "x",
        "--output-dir",
        "results/hotpotqa/04_experiments/correlation",
    ]


def test_experiments_passthrough_keeps_explicit_output_dir(monkeypatch):
    captured = {}

    def fake_run_correlation(cmd_args):
        captured["cmd_args"] = cmd_args
        return 0

    monkeypatch.setattr(cli, "_run_correlation_experiment", fake_run_correlation)

    args = SimpleNamespace(
        name="correlation",
        extra=["--dummy-flag", "x", "--output-dir", "custom/out"],
    )
    cfg = {"experiment_id": "hotpotqa", "workflow_type": "hotpotqa"}

    assert cli.cmd_experiments(args, cfg) == 0
    assert captured["cmd_args"] == ["--dummy-flag", "x", "--output-dir", "custom/out"]


def test_experiments_correlation_errors_if_canonical_latency_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    exp = "hotpotqa"
    workflow_dir = tmp_path / "results" / exp / "03_test"
    workflow_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(name="correlation", extra=[])
    cfg = {"experiment_id": exp, "workflow_type": "hotpotqa"}

    with pytest.raises(SystemExit, match="latency_file not found at canonical path"):
        cli.cmd_experiments(args, cfg)


def test_experiments_correlation_optional_config_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    workflow_dir = tmp_path / "custom" / "corr_data"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    latency_file = tmp_path / "results" / "hotpotqa" / "01_profile" / "latency_benchmark.json"
    latency_file.parent.mkdir(parents=True, exist_ok=True)
    latency_file.write_text("{}", encoding="utf-8")

    captured = {}

    def fake_run_correlation(cmd_args):
        captured["cmd_args"] = cmd_args
        return 0

    monkeypatch.setattr(cli, "_run_correlation_experiment", fake_run_correlation)

    args = SimpleNamespace(name="correlation", extra=[])
    cfg = {
        "experiment_id": "hotpotqa",
        "workflow_type": "hotpotqa",
        "correlation_workflow_all_results_dir": str(workflow_dir),
        "correlation_latency_file": "results/hotpotqa/01_profile/latency_benchmark.json",
        "correlation_output_dir": "outputs/correlation",
        "correlation_optimize_calibration": True,
    }

    assert cli.cmd_experiments(args, cfg) == 0
    assert captured["cmd_args"] == [
        "--workflow-all-results-dir",
        str(workflow_dir),
        "--workflow-type",
        "hotpotqa",
        "--latency-file",
        "results/hotpotqa/01_profile/latency_benchmark.json",
        "--output-dir",
        "outputs/correlation",
        "--optimize-calibration",
    ]


def test_experiments_correlation_rejects_noncanonical_latency_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    exp = "hotpotqa"
    workflow_dir = tmp_path / "results" / exp / "03_test"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    canonical_latency = tmp_path / "results" / exp / "01_profile" / "latency_benchmark.json"
    canonical_latency.parent.mkdir(parents=True, exist_ok=True)
    canonical_latency.write_text("{}", encoding="utf-8")
    noncanonical_latency = tmp_path / "custom" / "latency.json"
    noncanonical_latency.parent.mkdir(parents=True, exist_ok=True)
    noncanonical_latency.write_text("{}", encoding="utf-8")

    args = SimpleNamespace(name="correlation", extra=[])
    cfg = {
        "experiment_id": exp,
        "workflow_type": "hotpotqa",
        "correlation_latency_file": str(noncanonical_latency),
    }

    with pytest.raises(SystemExit, match="must be the canonical path"):
        cli.cmd_experiments(args, cfg)


@pytest.mark.parametrize(
    "name",
    [
        "unified-eval",
        "ablation",
        "average-alpha",
        "pareto-plot",
        "analysis-pipeline",
    ],
)
def test_cli_main_parser_rejects_removed_experiment_names(monkeypatch, name):
    monkeypatch.setattr("sys.argv", ["flowcompile", "experiments", name])
    with pytest.raises(SystemExit, match="invalid choice"):
        cli.main()


def test_correlation_module_import_smoke():
    from workflow_compiler.experiments import correlation

    assert callable(correlation.main)
    assert "workflow_compiler/experiments/correlation.py" in correlation.__file__.replace("\\", "/")
