from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from workflow_compiler.benchmarks import (
    get_benchmark,
    get_benchmark_info,
    list_benchmarks,
    register_benchmark,
)
from workflow_compiler.benchmarks import registry as benchmark_registry
from workflow_compiler.compiler.validation import _dataset_to_workflow_type, _metric_for_dataset


@pytest.fixture()
def registry_state_guard():
    snapshot = {
        "registry": dict(benchmark_registry._REGISTRY),
        "canonical": dict(benchmark_registry._CANONICAL_NAME),
        "aliases": {k: list(v) for k, v in benchmark_registry._ALIASES.items()},
        "alias_to_key": dict(benchmark_registry._ALIAS_TO_KEY),
        "discovered": benchmark_registry._DISCOVERED,
    }
    yield
    benchmark_registry._REGISTRY.clear()
    benchmark_registry._REGISTRY.update(snapshot["registry"])
    benchmark_registry._CANONICAL_NAME.clear()
    benchmark_registry._CANONICAL_NAME.update(snapshot["canonical"])
    benchmark_registry._ALIASES.clear()
    benchmark_registry._ALIASES.update(snapshot["aliases"])
    benchmark_registry._ALIAS_TO_KEY.clear()
    benchmark_registry._ALIAS_TO_KEY.update(snapshot["alias_to_key"])
    benchmark_registry._DISCOVERED = snapshot["discovered"]


def _install_temp_benchmark_module(tmp_path: Path, module_name: str) -> tuple:
    module_code = textwrap.dedent(
        """
        from typing import Any, Dict, List, Tuple

        from workflow_compiler.benchmarks import register_benchmark
        from workflow_compiler.benchmarks.benchmark import BaseBenchmark


        @register_benchmark()
        class TempRegistryBenchmark(BaseBenchmark):
            BENCHMARK_NAME = "TempRegistryBenchmark"
            ALIASES = ["temp_registry_benchmark", "tempreg"]
            WORKFLOW_TYPE = "math"
            METRIC_NAME = "accuracy"
            DEFAULT_SPLIT_PATHS = {
                "validate": "data/ours/temp_registry_validate.jsonl",
                "test": "data/ours/temp_registry_test.jsonl",
            }

            def __init__(self, name: str, file_path: str, log_path: str):
                super().__init__(name=name, file_path=file_path, log_path=log_path)

            async def evaluate_problem(self, problem: Dict[str, Any], workflow: Any) -> Tuple[str, str, str, float]:
                return "q", "p", "e", 1.0

            def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
                return 1.0, prediction

            def get_result_columns(self) -> List[str]:
                return ["question", "prediction", "expected_output", "score"]
        """
    )
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(module_code, encoding="utf-8")

    package = importlib.import_module("workflow_compiler.benchmarks")
    package_path = str(tmp_path)
    package.__path__.append(package_path)
    return package, package_path


def test_register_benchmark_reexport():
    assert callable(register_benchmark)


def test_registry_lists_builtin_benchmarks():
    rows = list_benchmarks(detailed=True)
    names = {row["name"] for row in rows}
    assert {"MATH", "GSM8K", "HotpotQA", "LiveCodeBench"}.issubset(names)

    info = get_benchmark_info("math500")
    assert info["name"] == "MATH"
    assert info["workflow_type"] == "math"
    assert info["metric_name"] == "accuracy"


def test_validation_uses_registry_for_aliases():
    assert _dataset_to_workflow_type("math500") == "math"
    assert _metric_for_dataset("math500") == "accuracy"


def test_registry_auto_discovers_temp_module_and_can_instantiate(tmp_path: Path, registry_state_guard):
    module_name = "temp_registry_benchmark_module"
    package, package_path = _install_temp_benchmark_module(tmp_path, module_name)

    try:
        benchmark_registry._DISCOVERED = False
        benchmark_registry.discover_benchmarks(force=True)

        info = get_benchmark_info("tempreg")
        assert info["name"] == "TempRegistryBenchmark"
        assert info["workflow_type"] == "math"
        assert info["metric_name"] == "accuracy"

        benchmark = get_benchmark(
            "tempreg",
            name=info["name"],
            file_path=str(tmp_path / "dummy.jsonl"),
            log_path=str(tmp_path),
        )
        assert benchmark.BENCHMARK_NAME == "TempRegistryBenchmark"
    finally:
        if package_path in package.__path__:
            package.__path__.remove(package_path)
        sys.modules.pop(f"workflow_compiler.benchmarks.{module_name}", None)
