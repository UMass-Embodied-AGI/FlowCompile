# -*- coding: utf-8 -*-
"""
GSM8K Benchmark - Grade School Math word problems.

GSM8K uses the same data format and evaluation logic as MATH/MATH500,
so this is simply an alias for MATHBenchmark.
"""

from workflow_compiler.benchmarks.math import MATHBenchmark
from workflow_compiler.benchmarks.registry import register_benchmark


# GSM8K uses the same benchmark class as MATH since they share the same format
@register_benchmark()
class GSM8KBenchmark(MATHBenchmark):
    """
    GSM8K Benchmark class.
    
    This is an alias for MATHBenchmark as GSM8K shares the same data format:
    - problem: The math problem text
    - solution: Step-by-step solution
    - answer: The final numerical answer
    - unique_id: Problem identifier
    """
    BENCHMARK_NAME = "GSM8K"
    ALIASES = ["gsm8k", "GSM8K"]
    WORKFLOW_TYPE = "gsm8k"
    METRIC_NAME = "accuracy"
    DEFAULT_SPLIT_PATHS = {
        "validate": "data/gsm8k_validate.jsonl",
        "test": "data/gsm8k_test.jsonl",
    }


__all__ = ["GSM8KBenchmark"]
