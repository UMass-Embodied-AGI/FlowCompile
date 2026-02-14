from workflow_compiler.compiler.profiling import get_experiment_config


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
