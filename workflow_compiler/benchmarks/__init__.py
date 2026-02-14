# -*- coding: utf-8 -*-
"""
Benchmark datasets for FlowCompile.

This module provides benchmark implementations for core tasks:
- MATH: Mathematical problem solving
- HotpotQA: Multi-hop question answering
- GSM8K: Grade school math problems
- LiveCodeBench: Code generation benchmarks
"""

from .benchmark import BaseBenchmark
from .math import MATHBenchmark
from .hotpotqa import HotpotQABenchmark
from .gsm8k import GSM8KBenchmark
from .livecodebench import LiveCodeBench, evaluate_generations_by_problem
from .registry import (
    discover_benchmarks,
    get_benchmark,
    get_benchmark_class,
    get_benchmark_info,
    list_benchmarks,
    register_benchmark,
)

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
]
