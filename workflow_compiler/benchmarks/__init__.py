"""Benchmark datasets for FlowCompile.

The package exposes benchmark classes and registry helpers, but avoids eager
imports so lightweight consumers such as the Sphinx docs build can import the
registry without pulling in the full benchmark runtime dependency set.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .benchmark import BaseBenchmark
    from .gsm8k import GSM8KBenchmark
    from .hotpotqa import HotpotQABenchmark
    from .livecodebench import LiveCodeBench, evaluate_generations_by_problem
    from .math import MATHBenchmark
    from .registry import (
        discover_benchmarks,
        get_benchmark,
        get_benchmark_class,
        get_benchmark_info,
        list_benchmarks,
        register_benchmark,
    )


_EXPORTS = {
    "BaseBenchmark": (".benchmark", "BaseBenchmark"),
    "MATHBenchmark": (".math", "MATHBenchmark"),
    "HotpotQABenchmark": (".hotpotqa", "HotpotQABenchmark"),
    "GSM8KBenchmark": (".gsm8k", "GSM8KBenchmark"),
    "LiveCodeBench": (".livecodebench", "LiveCodeBench"),
    "evaluate_generations_by_problem": (
        ".livecodebench",
        "evaluate_generations_by_problem",
    ),
    "register_benchmark": (".registry", "register_benchmark"),
    "discover_benchmarks": (".registry", "discover_benchmarks"),
    "get_benchmark": (".registry", "get_benchmark"),
    "get_benchmark_class": (".registry", "get_benchmark_class"),
    "get_benchmark_info": (".registry", "get_benchmark_info"),
    "list_benchmarks": (".registry", "list_benchmarks"),
    "DEFAULT_LATENCY_PROMPT_SOURCE": (".prompts", "DEFAULT_LATENCY_PROMPT_SOURCE"),
    "DEFAULT_LATENCY_PROMPT_TEXT": (".prompts", "DEFAULT_LATENCY_PROMPT_TEXT"),
    "get_default_latency_prompt_text": (".prompts", "get_default_latency_prompt_text"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))

__all__ = [
    "BaseBenchmark",
    "MATHBenchmark",
    "HotpotQABenchmark",
    "GSM8KBenchmark",
    "LiveCodeBench",
    "evaluate_generations_by_problem",
    "register_benchmark",
    "discover_benchmarks",
    "get_benchmark",
    "get_benchmark_class",
    "get_benchmark_info",
    "list_benchmarks",
    "DEFAULT_LATENCY_PROMPT_SOURCE",
    "DEFAULT_LATENCY_PROMPT_TEXT",
    "get_default_latency_prompt_text",
]
