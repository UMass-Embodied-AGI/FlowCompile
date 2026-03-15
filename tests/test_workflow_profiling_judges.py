from workflow_compiler.workflows.dsl_registry import get_workflow_module


def test_math_workflow_exposes_profiling_judges():
    judges = get_workflow_module("math").get_profiling_judges()
    assert set(judges) == {
        "programmer",
        "detailed_solver",
        "generate_solver",
        "refine_solver",
        "sc_ensemble",
    }


def test_hotpotqa_workflow_exposes_profiling_judges():
    judges = get_workflow_module("hotpotqa").get_profiling_judges()
    assert set(judges) == {"answer_generate", "format_answer", "sc_ensemble"}


def test_livecodebench_workflow_exposes_profiling_judges():
    judges = get_workflow_module("livecodebench").get_profiling_judges()
    assert set(judges) == {"code_generate", "reflection_test", "sc_ensemble"}
