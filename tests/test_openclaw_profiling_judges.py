from __future__ import annotations

import asyncio
from pathlib import Path

from workflow_compiler.compiler import profiling


class _FakeJudgeLLM:
    def __init__(self, response: str = "CORRECT"):
        self.response = response
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response

    async def aclose(self) -> None:
        return None


def _configure_openclaw_judge(monkeypatch, *, policies, schemas, response: str = "CORRECT"):
    fake_llm = _FakeJudgeLLM(response=response)
    monkeypatch.setattr(profiling, "create_llm_instance", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(profiling.BenchmarkConfig, "OPENCLAW_AGENT_POLICIES", policies)
    monkeypatch.setattr(profiling.BenchmarkConfig, "OPENCLAW_AGENT_SCHEMAS", schemas)
    monkeypatch.setattr(profiling.BenchmarkConfig, "WORKFLOW_TYPE", "openclaw_lobster")
    monkeypatch.setattr(profiling.BenchmarkConfig, "WORKFLOW_JUDGES", {})
    return fake_llm, profiling.JudgeEvaluator()


def test_openclaw_semantic_judge_accepts_boolean_required_field(monkeypatch):
    policies = {
        "select_efficiency": {
            "required_fields": ("selected", "reason"),
            "mode": "semantic_llm",
            "prompt": "Pred {predicted_json}\nGT {ground_truth_json}",
        }
    }
    schemas = {
        "select_efficiency": {
            "path": Path("/tmp/select_efficiency.schema.json"),
            "schema": {
                "type": "object",
                "required": ["selected", "reason"],
                "properties": {
                    "selected": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    }
    fake_llm, judge = _configure_openclaw_judge(monkeypatch, policies=policies, schemas=schemas)
    context = judge.build_context(
        ground_truth='{"selected": true, "reason": "efficiency"}',
        model_output='{"selected": true, "reason": "matched"}',
        agent_name="select_efficiency",
        input_prompt="Decide whether this is an efficiency paper.",
    )

    result = asyncio.run(judge.evaluate_context(context))

    assert result.is_correct is True
    assert len(fake_llm.prompts) == 1
    assert '"selected": true' in fake_llm.prompts[0]


def test_openclaw_semantic_judge_accepts_array_required_field(monkeypatch):
    policies = {
        "summarize_pdfs": {
            "required_fields": ("benchmarks", "core_method"),
            "mode": "semantic_llm",
            "prompt": "Pred {predicted_json}\nGT {ground_truth_json}",
        }
    }
    schemas = {
        "summarize_pdfs": {
            "path": Path("/tmp/summarize_pdfs.schema.json"),
            "schema": {
                "type": "object",
                "required": ["benchmarks", "core_method", "experimental_setup", "main_results"],
                "properties": {
                    "benchmarks": {"type": "array", "items": {"type": "string"}},
                    "core_method": {"type": "string"},
                    "experimental_setup": {"type": "string"},
                    "main_results": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    }
    fake_llm, judge = _configure_openclaw_judge(monkeypatch, policies=policies, schemas=schemas)
    context = judge.build_context(
        ground_truth='{"benchmarks": ["A", "B"], "core_method": "m", "experimental_setup": "e", "main_results": "r"}',
        model_output='{"benchmarks": ["A", "B"], "core_method": "m2", "experimental_setup": "e", "main_results": "r"}',
        agent_name="summarize_pdfs",
        input_prompt="Summarize the paper.",
    )

    result = asyncio.run(judge.evaluate_context(context))

    assert result.is_correct is True
    assert len(fake_llm.prompts) == 1
    assert '"benchmarks": ["A", "B"]' in fake_llm.prompts[0]


def test_openclaw_semantic_judge_rejects_schema_invalid_payload_before_judge_call(monkeypatch):
    policies = {
        "select_efficiency": {
            "required_fields": ("selected", "reason"),
            "mode": "semantic_llm",
            "prompt": "Pred {predicted_json}\nGT {ground_truth_json}",
        }
    }
    schemas = {
        "select_efficiency": {
            "path": Path("/tmp/select_efficiency.schema.json"),
            "schema": {
                "type": "object",
                "required": ["selected", "reason"],
                "properties": {
                    "selected": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    }
    fake_llm, judge = _configure_openclaw_judge(monkeypatch, policies=policies, schemas=schemas)
    context = judge.build_context(
        ground_truth='{"selected": true, "reason": "efficiency"}',
        model_output='{"selected": "true", "reason": "wrong type"}',
        agent_name="select_efficiency",
        input_prompt="Decide whether this is an efficiency paper.",
    )

    result = asyncio.run(judge.evaluate_context(context))

    assert result.is_correct is False
    assert fake_llm.prompts == []


def test_openclaw_strict_exact_normalizes_strings_but_keeps_typed_comparison(monkeypatch):
    policies = {
        "classify": {
            "required_fields": ("category",),
            "mode": "strict_exact",
        }
    }
    schemas = {
        "classify": {
            "path": Path("/tmp/classify.schema.json"),
            "schema": {
                "type": "object",
                "required": ["category"],
                "properties": {
                    "category": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    }
    _, judge = _configure_openclaw_judge(monkeypatch, policies=policies, schemas=schemas)
    context = judge.build_context(
        ground_truth='{"category": "reply_required"}',
        model_output='{"category": " REPLY_REQUIRED "}',
        agent_name="classify",
        input_prompt="Classify the email.",
    )

    result = asyncio.run(judge.evaluate_context(context))

    assert result.is_correct is True
