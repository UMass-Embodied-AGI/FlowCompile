# Adding a New Python DSL Workflow

This project uses a PyTorch-like DSL to define workflows as Python code. Each
workflow lives in its own folder under `workflow_compiler/workflows/` and is
loaded through the DSL registry.

## 1) Create a workflow folder

If you’re adding a brand‑new workflow type, create a folder:

```
workflow_compiler/workflows/<workflow_name>/
```

Tip: You can copy the template at `workflow_compiler/workflows/template/` to get started.

Typical contents:

- `workflow.py` – the Python DSL workflow definition (required)
- `agents.py` – optional, only if you need custom SubAgent implementations
- `judges.py` – optional, used when profiling needs workflow-owned agent correctness logic
- `__init__.py` – export your DSL class for convenience

If you add `agents.py`, implement custom agents by subclassing `SubAgent` and
only defining `run(...) -> AgentResult`. Do not build metadata manually:
`SubAgent.execute` auto-records `agent`, `call_number`, token counts, raw LLM
prompt/output, processed output, timestamp, and status.

If you add `judges.py`, expose `get_profiling_judges()` and register one async
judge per inferred agent name that needs custom profiling evaluation. This is
the extension point for agent-specific correctness logic; do not add new
hardcoded branches in `compiler/profiling.py`.

## 2) Implement the DSL workflow (`workflow.py`)

Example skeleton:

```python
from workflow_compiler.dsl.torchlike import WorkflowModule, AgentNode, ToolNode

class MyWorkflowDSL(WorkflowModule):
    workflow_type = "myworkflow"

    def __init__(self):
        super().__init__(name="myworkflow_dsl", execution_mode="sequential")
        self.solver = AgentNode("solver")
        self.extract = ToolNode("extract", impl="extract_answer")

    # Keep this signature aligned across workflows.
    def forward(self, query):
        problem = query["problem"]
        solution = self.solver(problem=problem)
        answer = self.extract(solution=solution)
        return {
            "final_answer": answer,
            "full_solution": solution,
            "final_solution": solution,
        }
```

Guidelines:

- Use normal Python grammar (loops, breaks, etc.).
- Do **not** add any evaluation logic or correctness checks inside the workflow.
- Put profiling correctness logic in `judges.py`, keyed by canonical `AgentNode.name`.
- Auto-backward is enabled by default. You only need to implement `backward(payload)` if
  your workflow uses unsupported conditional logic or custom composition rules.
- `AgentNode.name` is the canonical key end-to-end (structure IDs, metrics payload, runtime config keys).
- Do not define alias maps (`metric_agents`, `profiling_agents`, `setting_aliases`, `runtime_agent_map`, `subagent_aliases`).
- Keep `execution_mode="sequential"` for now. Additional modes may be added later.
- Keep a consistent I/O surface across workflows:
  - Input: single `query` object
  - Query keys: `problem`, `entry_point`, `question_id`
  - Outputs: `final_answer`, `full_solution`, `final_solution`
- If you provide a manual backward, use a single `backward(payload)` argument:
  - `payload["structure"]`: selected structure dict
  - `payload["metrics"]`: dict of per-agent metric DataFrames
- Do not hand-write structure enumerators/config counters:
  - `enumerate_structures()` is inferred automatically from `forward`.
  - `get_full_structure()` returns the full inferred structure.
- Profiling/prediction strictly uses inferred `AgentNode.name` keys:
  - all inferred agents must have profiling data when generating configs.
- Keep outputs consistent with the trace builder (see step 4).

## 3) Register the workflow

Add your workflow to the DSL registry:

`workflow_compiler/workflows/dsl_registry.py`

```python
from workflow_compiler.workflows.myworkflow.workflow import MyWorkflowDSL

if workflow_type == "myworkflow":
    return MyWorkflowDSL()
```

## 4) Update DSL runtime preprocess + trace building (if needed)

If your workflow has different inputs or outputs, update:

- `_preprocess_query` in `workflow_compiler/dsl/runtime.py`
- `_build_trace_*` in `workflow_compiler/dsl/runtime.py`

These functions control how raw dataset rows become workflow inputs and how
final outputs are written to `trace.jsonl`.

## 5) (Optional) Export in the package `__init__.py`

Add your DSL class and judge registry helper to the workflow package’s `__init__.py` for convenience:

```python
from .judges import get_profiling_judges
from .workflow import MyWorkflowDSL
__all__ = ["MyWorkflowDSL", "get_profiling_judges", ...]
```

## 6) Quick sanity check

Run a small workflow on one sample to confirm:

- The DSL compiles and executes without errors.
- The trace output contains final answers/solutions only.
- Evaluation adds `score` and `metric` later (validation step).

---

That’s it — you can now add new workflows without touching any centralized
workflow-definition shim.
