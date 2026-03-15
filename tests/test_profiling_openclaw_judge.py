import asyncio

import pytest

from workflow_compiler.compiler import profiling
from workflow_compiler.workflows.dsl_registry import get_workflow_module


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


def _set_openclaw_policies(monkeypatch):
    monkeypatch.setattr(
        profiling.BenchmarkConfig,
        "OPENCLAW_AGENT_POLICIES",
        {
            "summarize_each": {
                "required_fields": ("summary",),
                "mode": "semantic_llm",
                "prompt": "GT: {ground_truth_field}\nPred: {predicted_field}\nInput: {input_prompt}",
            },
            "classify": {
                "required_fields": ("category",),
                "mode": "strict_exact",
            },
            "overview": {
                "required_fields": ("overview_paragraph",),
                "mode": "semantic_llm",
                "prompt": "GT: {ground_truth_field}\nPred: {predicted_field}",
            },
            "ask_questions": {
                "required_fields": ("question",),
                "mode": "semantic_llm",
                "prompt": "GT: {ground_truth_field}\nPred: {predicted_field}",
            },
            "draft_replies": {
                "required_fields": ("draft_body",),
                "mode": "semantic_llm",
                "prompt": "GT: {ground_truth_field}\nPred: {predicted_field}\nInput: {input_prompt}",
            },
        },
    )


def _set_workflow_judges(monkeypatch, workflow_type: str):
    monkeypatch.setattr(
        profiling.BenchmarkConfig,
        "WORKFLOW_JUDGES",
        get_workflow_module(workflow_type).get_profiling_judges(),
    )


def test_openclaw_uses_configured_policy_map(monkeypatch):
    judge = _make_judge(response="CORRECT")
    monkeypatch.setattr(
        profiling.BenchmarkConfig,
        "OPENCLAW_AGENT_POLICIES",
        {
            "triage": {
                "required_fields": ("decision",),
                "mode": "strict_exact",
            }
        },
    )

    ok = asyncio.run(
        judge.evaluate(
            ground_truth='{"decision":"reply"}',
            model_output='{"decision":" reply "}',
            agent_name="triage",
            input_prompt="triage prompt",
        )
    )

    assert ok is True
    assert judge.judge_llm.prompts == []


def test_openclaw_semantic_prompt_supports_json_placeholders(monkeypatch):
    judge = _make_judge(response="CORRECT")
    monkeypatch.setattr(
        profiling.BenchmarkConfig,
        "OPENCLAW_AGENT_POLICIES",
        {
            "planner": {
                "required_fields": ("subject", "body"),
                "mode": "semantic_llm",
                "prompt": "Input: {input_prompt}\nFields: {required_fields}\nGT: {ground_truth_json}\nPred: {predicted_json}",
            }
        },
    )

    ok = asyncio.run(
        judge.evaluate(
            ground_truth='{"subject":"Status","body":"Please ship it"}',
            model_output='{"subject":"Status","body":"Please ship it today"}',
            agent_name="planner",
            input_prompt="Write a subject and body",
        )
    )

    assert ok is True
    assert len(judge.judge_llm.prompts) == 1
    prompt = judge.judge_llm.prompts[0]
    assert "subject" in prompt
    assert "body" in prompt
    assert '"Status"' in prompt


def test_openclaw_classify_strict_exact_match_normalized():
    profiling.BenchmarkConfig.OPENCLAW_AGENT_POLICIES = {
        "classify": {"required_fields": ("category",), "mode": "strict_exact"}
    }
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
    profiling.BenchmarkConfig.OPENCLAW_AGENT_POLICIES = {
        "classify": {"required_fields": ("category",), "mode": "strict_exact"}
    }
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
    profiling.BenchmarkConfig.OPENCLAW_AGENT_POLICIES = {
        "summarize_each": {
            "required_fields": ("summary",),
            "mode": "semantic_llm",
            "prompt": "GT: {ground_truth_field}\nPred: {predicted_field}",
        }
    }
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
    profiling.BenchmarkConfig.OPENCLAW_AGENT_POLICIES = {
        agent_name: {
            "required_fields": (field_name,),
            "mode": "semantic_llm",
            "prompt": "GT: {ground_truth_field}\nPred: {predicted_field}",
        }
    }
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
    profiling.BenchmarkConfig.OPENCLAW_AGENT_POLICIES = {
        "summarize_each": {
            "required_fields": ("summary",),
            "mode": "semantic_llm",
            "prompt": "GT: {ground_truth_field}\nPred: {predicted_field}",
        }
    }
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
    profiling.BenchmarkConfig.OPENCLAW_AGENT_POLICIES = {
        "draft_replies": {
            "required_fields": ("draft_body",),
            "mode": "semantic_llm",
            "prompt": "Input: {input_prompt}\nGT: {ground_truth_field}\nPred: {predicted_field}",
        }
    }
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


def test_openclaw_without_configured_policies_falls_back_to_non_openclaw_paths():
    profiling.BenchmarkConfig.OPENCLAW_AGENT_POLICIES = None
    profiling.BenchmarkConfig.WORKFLOW_JUDGES = {}
    judge = _make_judge(response="CORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth='{"category":"newsletter"}',
            model_output='{"category":"newsletter"}',
            agent_name="classify",
            input_prompt="classify prompt",
        )
    )

    assert ok is False

def test_judge_evaluator_uses_profile_endpoint_role(monkeypatch):
    calls = []

    monkeypatch.setattr(profiling.JudgeEvaluator, "_load_livecodebench_test_cache", lambda self: None)

    def fake_create_llm_instance(model_name, endpoint_role=None):
        calls.append((model_name, endpoint_role))
        return _DummyJudgeLLM()

    monkeypatch.setattr(profiling, "create_llm_instance", fake_create_llm_instance)

    judge = profiling.JudgeEvaluator(judge_model="judge-model")

    assert isinstance(judge.judge_llm, _DummyJudgeLLM)
    assert calls == [("judge-model", "profile")]


def test_agent_benchmarker_uses_profile_endpoint_role(monkeypatch):
    calls = []

    def fake_create_llm_instance(model_name, endpoint_role=None):
        calls.append((model_name, endpoint_role))
        return _DummyJudgeLLM()

    monkeypatch.setattr(profiling, "create_llm_instance", fake_create_llm_instance)

    benchmarker = profiling.AgentBenchmarker(agent_name="triage", judge=object())
    llm = benchmarker._get_llm("qwen35-9b-awq")

    assert isinstance(llm, _DummyJudgeLLM)
    assert benchmarker._get_llm("qwen35-9b-awq") is llm
    assert calls == [("qwen35-9b-awq", "profile")]


def test_answer_generate_uses_configured_judge_prompt(monkeypatch):
    _set_workflow_judges(monkeypatch, "hotpotqa")
    judge = _make_judge(response="CORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth="<answer>Paris</answer>",
            model_output="<answer>Paris</answer>",
            agent_name="answer_generate",
            question="What is the capital of France?",
        )
    )

    assert ok is True
    assert "What is the capital of France?" in judge.judge_llm.prompts[0]


def test_programmer_uses_workflow_owned_judge(monkeypatch):
    _set_workflow_judges(monkeypatch, "math")
    judge = _make_judge(response="CORRECT")
    monkeypatch.setattr(
        profiling.JudgeEvaluator,
        "execute_code_in_subprocess",
        lambda self, code, timeout_seconds=5.0: asyncio.sleep(0, result=("Success", "42")),
    )

    ok = asyncio.run(
        judge.evaluate(
            ground_truth="42",
            model_output="```python\nprint(42)\n```",
            agent_name="programmer",
        )
    )

    assert ok is True
    assert "Actual Execution Output" in judge.judge_llm.prompts[0]


def test_format_answer_returns_metric_from_workflow_judge(monkeypatch):
    _set_workflow_judges(monkeypatch, "hotpotqa")
    judge = _make_judge(response="CORRECT")

    result = asyncio.run(
        judge.evaluate_context(
            judge.build_context(
                ground_truth="Paris",
                model_output="Paris",
                agent_name="format_answer",
                workflow_type="hotpotqa",
            )
        )
    )

    assert result.metric_name == "f1_score"
    assert result.metric_value == 1.0
    assert result.is_correct == 1.0
    assert judge.judge_llm.prompts == []


def test_livecodebench_workflow_judge_uses_private_tests(monkeypatch):
    _set_workflow_judges(monkeypatch, "livecodebench")
    judge = _make_judge(response="CORRECT")
    observed = {}

    async def fake_evaluate_code_with_private_tests(code, original_sample):
        observed["code"] = code
        observed["sample"] = original_sample
        return True

    monkeypatch.setattr(judge, "evaluate_code_with_private_tests", fake_evaluate_code_with_private_tests)

    ok = asyncio.run(
        judge.evaluate(
            ground_truth="unused",
            model_output="```python\ndef solve():\n    return 1\n```",
            agent_name="code_generate",
            workflow_type="livecodebench",
            original_sample={"question_id": "q1"},
        )
    )

    assert ok is True
    assert "def solve()" in observed["code"]
    assert observed["sample"] == {"question_id": "q1"}
    assert judge.judge_llm.prompts == []


@pytest.mark.parametrize(
    ("workflow_type", "problem", "expected_text"),
    [
        ("math", "2+2?", "Ground Truth Solution"),
        ("hotpotqa", "Question: Who wrote Hamlet?\nAnswer:", "Question:"),
        ("livecodebench", "Write solve()", "Problem:"),
    ],
)
def test_sc_ensemble_uses_workflow_local_prompt_shape(monkeypatch, workflow_type, problem, expected_text):
    _set_workflow_judges(monkeypatch, workflow_type)
    judge = _make_judge(response="CORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth="\\boxed{A}",
            model_output="\\boxed{A}",
            agent_name="sc_ensemble",
            problem=problem,
            solutions=["candidate A", "candidate B"],
            workflow_type=workflow_type,
        )
    )

    assert ok is True
    assert expected_text in judge.judge_llm.prompts[0]


def test_missing_workflow_judge_fails_cleanly(capsys):
    profiling.BenchmarkConfig.WORKFLOW_JUDGES = {}
    judge = _make_judge(response="CORRECT")

    ok = asyncio.run(
        judge.evaluate(
            ground_truth="42",
            model_output="42",
            agent_name="detailed_solver",
        )
    )

    assert ok is False
    assert "[judge:judge_error] agent=detailed_solver missing workflow judge" in capsys.readouterr().out
