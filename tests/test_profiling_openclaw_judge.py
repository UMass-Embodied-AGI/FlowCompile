import asyncio

import pytest

from workflow_compiler.compiler import profiling


class _DummyJudgeLLM:
    def __init__(self, response: str = "CORRECT"):
        self.response = response
        self.prompts = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _make_judge(response: str = "CORRECT"):
    judge = object.__new__(profiling.JudgeEvaluator)
    judge.judge_llm = _DummyJudgeLLM(response=response)
    judge.livecodebench_test_cache = {}
    return judge


def test_openclaw_classify_strict_exact_match_normalized():
    judge = _make_judge(response="INCORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth='{"category":"newsletter"}',
            model_output='{"category":" NewsLetter "}',
            agent_name="classify",
            input_prompt="classify prompt",
        )
    )

    assert ok is True
    assert judge.judge_llm.prompts == []


def test_openclaw_classify_strict_mismatch_logs_reason(capsys):
    judge = _make_judge(response="CORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth='{"category":"newsletter"}',
            model_output='{"category":"finance"}',
            agent_name="classify",
            input_prompt="classify prompt",
        )
    )

    assert ok is False
    assert judge.judge_llm.prompts == []
    assert "[judge:strict_mismatch] agent=classify" in capsys.readouterr().out


def test_openclaw_invalid_json_fails_before_llm(capsys):
    judge = _make_judge(response="CORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth='{"summary":"ok"}',
            model_output="not_json",
            agent_name="summarize_each",
            input_prompt="summary prompt",
        )
    )

    assert ok is False
    assert judge.judge_llm.prompts == []
    assert "[judge:invalid_json] agent=summarize_each" in capsys.readouterr().out


@pytest.mark.parametrize(
    "agent_name,field_name",
    [
        ("summarize_each", "summary"),
        ("overview", "overview_paragraph"),
        ("ask_questions", "question"),
    ],
)
def test_openclaw_semantic_agents_use_llm(agent_name: str, field_name: str):
    judge = _make_judge(response="CORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth=f'{{"{field_name}":"ground truth value"}}',
            model_output=f'{{"{field_name}":"predicted value"}}',
            agent_name=agent_name,
            input_prompt=f"{agent_name} prompt instructions",
        )
    )

    assert ok is True
    assert len(judge.judge_llm.prompts) == 1
    prompt = judge.judge_llm.prompts[0]
    assert "ground truth value" in prompt
    assert "predicted value" in prompt


def test_openclaw_semantic_agent_incorrect_from_judge_logs_reason(capsys):
    judge = _make_judge(response="INCORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth='{"summary":"ground truth value"}',
            model_output='{"summary":"predicted value"}',
            agent_name="summarize_each",
            input_prompt="summary prompt instructions",
        )
    )

    assert ok is False
    assert len(judge.judge_llm.prompts) == 1
    assert "[judge:judge_incorrect] agent=summarize_each" in capsys.readouterr().out


def test_openclaw_draft_prompt_includes_input_prompt_and_fields():
    judge = _make_judge(response="CORRECT")
    input_prompt = "Draft one plain-text reply body and follow all instruction constraints."
    gt_text = "Ground truth draft body"
    pred_text = "Predicted draft body"

    ok = asyncio.run(
        judge.evaluate(
            ground_truth=f'{{"draft_body":"{gt_text}"}}',
            model_output=f'{{"draft_body":"{pred_text}"}}',
            agent_name="draft_replies",
            input_prompt=input_prompt,
        )
    )

    assert ok is True
    assert len(judge.judge_llm.prompts) == 1
    prompt = judge.judge_llm.prompts[0]
    assert input_prompt in prompt
    assert gt_text in prompt
    assert pred_text in prompt
