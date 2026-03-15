import asyncio
import time

import pytest

from workflow_compiler.compiler import latency


class _DummyReporter:
    def __init__(self):
        self.warnings = []

    def child(self, _name: str):
        return self

    def warn(self, msg: str):
        self.warnings.append(msg)


def test_measure_batch_openai_visible_stream_keeps_stream_timing(monkeypatch):
    async def fake_stream(*_args, **_kwargs):
        start = time.perf_counter() + 0.5
        return latency.OpenAIStreamResult(
            request_start_s=start,
            first_visible_token_s=start + 0.2,
            end_s=start + 1.2,
            prompt_tokens=100,
            completion_tokens=50,
            saw_visible_delta=True,
        )

    async def fail_probe(*_args, **_kwargs):
        raise AssertionError("TTFT probe should not run for visible streams.")

    monkeypatch.setattr(latency, "_stream_one_openai", fake_stream)
    monkeypatch.setattr(latency, "_probe_openai_ttft", fail_probe)

    stats = asyncio.run(
        latency.measure_batch_openai(
            client=object(),
            request_model="test-model",
            prompt_text="hello",
            batch_size=1,
            max_new_tokens=32,
            seed=0,
        )
    )

    assert stats.total_generated_tokens == 50
    assert stats.decode_time_s == pytest.approx(1.0, abs=1e-6)
    assert stats.decode_tok_per_s == pytest.approx(50.0, abs=1e-6)


def test_measure_batch_openai_hidden_tokens_uses_probe_once(monkeypatch):
    reporter = _DummyReporter()
    monkeypatch.setattr(latency, "get_reporter", lambda: reporter)

    calls = {"stream": 0, "probe": 0}

    async def fake_stream(*_args, **_kwargs):
        calls["stream"] += 1
        start = time.perf_counter() + 0.5 + (0.1 * (calls["stream"] - 1))
        return latency.OpenAIStreamResult(
            request_start_s=start,
            first_visible_token_s=None,
            end_s=start + 3.0,
            prompt_tokens=10,
            completion_tokens=120,
            saw_visible_delta=False,
        )

    async def fake_probe(*_args, **_kwargs):
        calls["probe"] += 1
        return 0.8

    monkeypatch.setattr(latency, "_stream_one_openai", fake_stream)
    monkeypatch.setattr(latency, "_probe_openai_ttft", fake_probe)

    stats = asyncio.run(
        latency.measure_batch_openai(
            client=object(),
            request_model="hidden-model",
            prompt_text="hello",
            batch_size=2,
            max_new_tokens=32,
            seed=0,
        )
    )

    assert calls["stream"] == 2
    assert calls["probe"] == 1
    assert stats.total_generated_tokens == 240
    assert stats.decode_time_s > 0.0
    assert stats.decode_tok_per_s is not None
    assert len(reporter.warnings) == 1
    assert "TTFT probe fallback" in reporter.warnings[0]


def test_measure_batch_openai_zero_tokens_keeps_zero_decode(monkeypatch):
    async def fake_stream(*_args, **_kwargs):
        start = time.perf_counter() + 0.5
        return latency.OpenAIStreamResult(
            request_start_s=start,
            first_visible_token_s=None,
            end_s=start + 1.0,
            prompt_tokens=10,
            completion_tokens=0,
            saw_visible_delta=False,
        )

    async def fail_probe(*_args, **_kwargs):
        raise AssertionError("TTFT probe should not run when completion tokens are zero.")

    monkeypatch.setattr(latency, "_stream_one_openai", fake_stream)
    monkeypatch.setattr(latency, "_probe_openai_ttft", fail_probe)

    stats = asyncio.run(
        latency.measure_batch_openai(
            client=object(),
            request_model="test-model",
            prompt_text="hello",
            batch_size=1,
            max_new_tokens=32,
            seed=0,
        )
    )

    assert stats.total_generated_tokens == 0
    assert stats.decode_time_s == 0.0
    assert stats.decode_tok_per_s is None


def test_load_model_routes_uses_local_endpoint_for_latency_role(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "endpoints:",
                '  local_base_url: "http://127.0.0.1:4000"',
                '  profile_base_url: "http://profile-host:4000"',
                "models:",
                "  qwen35-9b-awq:",
                '    api_type: "openai"',
                '    api_key: "dummy"',
                '    hf_model_name: "QuantTrio/Qwen3.5-9B-AWQ"',
            ]
        ),
        encoding="utf-8",
    )

    routes = latency._load_model_routes(str(config_path), endpoint_role="latency")

    assert len(routes) == 1
    assert routes[0]["base_url"] == "http://127.0.0.1:4000"
    assert routes[0]["request_model"] == "qwen35-9b-awq"
    assert "QuantTrio/Qwen3.5-9B-AWQ" in routes[0]["aliases"]


def test_load_model_routes_keeps_legacy_base_url_without_endpoints(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  qwen35-9b-awq:",
                '    api_type: "openai"',
                '    base_url: "http://legacy-host:4000"',
                '    api_key: "dummy"',
                '    hf_model_name: "QuantTrio/Qwen3.5-9B-AWQ"',
            ]
        ),
        encoding="utf-8",
    )

    routes = latency._load_model_routes(str(config_path), endpoint_role="latency")

    assert len(routes) == 1
    assert routes[0]["base_url"] == "http://legacy-host:4000"
