from __future__ import annotations

from flowcompile.benchmarks.hotpotqa import HotpotQABenchmark


def test_hotpotqa_normalize_answer_handles_none():
    benchmark = HotpotQABenchmark(name="HotpotQA", file_path="data/hotpotqa_test.jsonl", log_path=".")
    assert benchmark.normalize_answer(None) == ""
