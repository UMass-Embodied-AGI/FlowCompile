import os

from workflow_compiler.compiler import latency


def test_configure_cuda_multiprocessing_sets_vllm_env(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    latency._configure_cuda_multiprocessing()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"


def test_run_latency_benchmark_configures_mp_for_non_openai(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("hello", encoding="utf-8")

    called = {"mp": 0}

    def fake_configure_mp():
        called["mp"] += 1

    monkeypatch.setattr(latency, "_configure_cuda_multiprocessing", fake_configure_mp)
    monkeypatch.setattr(latency, "_run_latency_benchmark_vllm", lambda **_: {})

    latency.run_latency_benchmark(
        models=["Qwen/Qwen3-4B"],
        output_json=str(tmp_path / "out.json"),
        prompt_file=str(prompt_file),
        backend="vllm",
    )
    assert called["mp"] == 1


def test_run_latency_benchmark_skips_mp_config_for_openai(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("hello", encoding="utf-8")

    called = {"mp": 0}

    def fake_configure_mp():
        called["mp"] += 1

    monkeypatch.setattr(latency, "_configure_cuda_multiprocessing", fake_configure_mp)
    monkeypatch.setattr(latency, "_run_latency_benchmark_openai", lambda **_: {})

    latency.run_latency_benchmark(
        models=["Qwen/Qwen3-4B"],
        output_json=str(tmp_path / "out.json"),
        prompt_file=str(prompt_file),
        backend="openai",
        model_config_path="configs/config.yaml",
    )
    assert called["mp"] == 0
