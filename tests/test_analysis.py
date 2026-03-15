"""
Unit tests for workflow_compiler.core.analysis module.
"""

from pathlib import Path

import pytest

from workflow_compiler.core.analysis import (
    calculate_latency,
    compute_pareto_frontier,
    extract_model_name,
    get_default_latency_data,
    get_hf_model_name,
)
from workflow_compiler.core.analysis.reporting import calculate_latency_from_trace


@pytest.fixture
def model_config_path(tmp_path: Path) -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  qwen3-4b:",
                "    hf_model_name: Qwen/Qwen3-4B",
                "  qwen3-8b-route:",
                "    model: qwen3-8b",
                "    hf_model_name: Qwen/Qwen3-8B",
                "  qwq-32b:",
                "    hf_model_name: Qwen/QwQ-32B",
                "  ds-32b:",
                "    hf_model_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            ]
        ),
        encoding="utf-8",
    )
    return str(config_path)


class TestModelNameMapping:
    """Tests for model name mapping functions."""

    def test_get_hf_model_name(self, model_config_path: str):
        assert get_hf_model_name("qwen3-4b", model_config_path=model_config_path) == "Qwen/Qwen3-4B"
        assert get_hf_model_name("qwen3-8b", model_config_path=model_config_path) == "Qwen/Qwen3-8B"
        assert get_hf_model_name("qwen3-8b-route", model_config_path=model_config_path) == "Qwen/Qwen3-8B"

    def test_get_hf_model_name_missing_alias_raises(self, model_config_path: str):
        with pytest.raises(ValueError, match="unknown"):
            get_hf_model_name("unknown", model_config_path=model_config_path)

    def test_extract_model_name_basic(self):
        assert extract_model_name("qwen3-4b") == "qwen3-4b"
        assert extract_model_name("qwen3-4b_budget_1000") == "qwen3-4b"
        assert extract_model_name("qwen3-8b_budget_unlimited") == "qwen3-8b"

    def test_extract_model_name_with_hf(self, model_config_path: str):
        assert (
            extract_model_name(
                "qwen3-4b",
                return_hf_name=True,
                model_config_path=model_config_path,
            )
            == "Qwen/Qwen3-4B"
        )
        assert (
            extract_model_name(
                "qwen3-4b_budget_1000",
                return_hf_name=True,
                model_config_path=model_config_path,
            )
            == "Qwen/Qwen3-4B"
        )

    def test_extract_model_name_with_budget(self):
        model, budget = extract_model_name("qwen3-4b_budget_1000", return_budget=True)
        assert model == "qwen3-4b"
        assert budget == 1000

        model, budget = extract_model_name("qwen3-8b_budget_unlimited", return_budget=True)
        assert model == "qwen3-8b"
        assert budget == -1

        model, budget = extract_model_name("qwen3-4b", return_budget=True)
        assert model == "qwen3-4b"
        assert budget == 0

    def test_extract_model_name_non_string(self):
        assert extract_model_name(True) == "True"
        assert extract_model_name(123) == "123"

        model, budget = extract_model_name(True, return_budget=True)
        assert model == "True"
        assert budget == 0

    def test_extract_model_name_local_alias_no_longer_supported(self, model_config_path: str):
        with pytest.raises(ValueError, match="qwen35-4b-local"):
            extract_model_name(
                "qwen35-4b-local_budget_1000",
                return_hf_name=True,
                model_config_path=model_config_path,
            )


class TestLatencyCalculation:
    """Tests for latency calculation functions."""

    def test_get_default_latency_data(self, model_config_path: str):
        data = get_default_latency_data(model_config_path=model_config_path)
        assert "Qwen/Qwen3-4B" in data
        assert "prefill_latency_per_token" in data["Qwen/Qwen3-4B"]
        assert "decode_latency_per_token" in data["Qwen/Qwen3-4B"]

    def test_calculate_latency(self, model_config_path: str):
        latency_data = {
            "Qwen/Qwen3-4B": {
                "prefill_latency_per_token": 0.0002,
                "decode_latency_per_token": 0.002,
            }
        }

        latency = calculate_latency(100, 50, "qwen3-4b", latency_data, model_config_path=model_config_path)
        expected = 100 * 0.0002 + 50 * 0.002
        assert abs(latency - expected) < 1e-6

        latency = calculate_latency(
            100,
            50,
            "qwen3-4b_budget_1000",
            latency_data,
            model_config_path=model_config_path,
        )
        assert abs(latency - expected) < 1e-6

    def test_calculate_latency_missing_alias_raises(self, model_config_path: str):
        with pytest.raises(ValueError, match="unknown_model"):
            calculate_latency(100, 50, "unknown_model", {}, model_config_path=model_config_path)

    def test_calculate_latency_missing_benchmark_entry_returns_zero(self, model_config_path: str):
        latency = calculate_latency(100, 50, "qwen3-4b", {}, model_config_path=model_config_path)
        assert latency == 0.0

    def test_calculate_latency_from_trace_uses_model_config_path(self, tmp_path: Path, model_config_path: str):
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            (
                '{"score": 1.0, "metric": "accuracy", "steps": ['
                '{"agent": "solver", "metadata": {"input_tokens": 100, "output_tokens": 50}}'
                "]}\n"
            ),
            encoding="utf-8",
        )
        latency_data = {
            "Qwen/Qwen3-4B": {
                "prefill_latency_per_token": 0.0002,
                "decode_latency_per_token": 0.002,
            }
        }

        result = calculate_latency_from_trace(
            trace_path,
            latency_data,
            {"solver": "qwen3-4b_budget_1000"},
            model_config_path=model_config_path,
        )

        assert abs(result["mean_latency"] - 0.12) < 1e-6


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_pareto_basic(self):
        points = [
            (1.0, 5.0),
            (2.0, 4.0),
            (3.0, 3.0),
            (4.0, 2.0),
            (2.5, 4.5),
        ]

        frontier = compute_pareto_frontier(points, maximize_x=False, maximize_y=True)

        assert len(frontier) >= 1
        assert (1.0, 5.0) in frontier

    def test_pareto_empty(self):
        assert compute_pareto_frontier([]) == []

    def test_pareto_single(self):
        points = [(1.0, 2.0)]
        assert compute_pareto_frontier(points) == points
