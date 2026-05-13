# Adding a Workflow

This guide documents the Python DSL workflow extension path used by
FlowCompile. A workflow should be a structured graph of reusable agents and
tools; evaluation and scoring belong in benchmarks, not in the workflow
definition.

## 1. Create a Workflow Package

Add a package under:

```text
workflow_compiler/workflows/<workflow_name>/
```

Typical contents:

- `workflow.py` for the DSL workflow definition
- `agents.py` for custom `SubAgent` implementations when needed
- `__init__.py` to re-export the workflow class

The repository includes a template package under `workflow_compiler/workflows/template/`.

If you add custom agents, subclass `SubAgent` and implement `run(...) ->
AgentResult`. Let `SubAgent.execute` record metadata such as token counts,
raw prompts, raw outputs, processed outputs, timestamps, and status.

## 2. Implement the DSL Workflow

A typical DSL workflow subclasses `WorkflowModule` and defines a
`forward(query)` method:

```python
from workflow_compiler.dsl.torchlike import WorkflowModule, AgentNode, ToolNode


class MyWorkflowDSL(WorkflowModule):
    workflow_type = "myworkflow"

    def __init__(self):
        super().__init__(name="myworkflow_dsl", execution_mode="sequential")
        self.solver = AgentNode("solver")
        self.extract = ToolNode("extract", impl="extract_answer")

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

## Workflow Guidelines

- Use standard Python control flow in the DSL. `range(...)` loops are unrolled
  during capture, and loop-break patterns of the form `if <cond>: break` are
  supported for tool-output fields such as `test_passed`.
- Do not embed evaluation logic in the workflow definition.
- Keep the input surface centered on a single `query` object.
- Use canonical `AgentNode` names. The node name is the key used by profiling
  data, compiled runtime configs, and structure IDs.
- Keep outputs aligned with the runtime trace builders:
  - `final_answer`
  - `full_solution`
  - `final_solution`
- Prefer inferred structure enumeration rather than hand-maintained enumerators.
- Keep `execution_mode="sequential"` unless the runtime adds explicit support for more modes.
- Do not define alias maps such as `metric_agents`, `profiling_agents`,
  `runtime_agent_map`, or `subagent_aliases`; the current implementation
  expects canonical agent names.

## 3. Register the Workflow

Update `workflow_compiler/workflows/dsl_registry.py` so the new `workflow_type` resolves to your DSL class.

The current flat CLI schema accepts the built-in workflow types `math`,
`gsm8k`, `hotpotqa`, and `livecodebench`. A genuinely new workflow type also
requires updating the CLI validator in `workflow_compiler/core/cli.py` and the
runtime support paths that dispatch by workflow type.

## 4. Let Auto-Backward Handle the Proxy

`WorkflowModule.backward(payload)` defaults to the auto-backward proxy in
`workflow_compiler.dsl.auto_backward`. It composes profiled sub-agent accuracy
and latency according to the captured graph and inferred structure.

Only implement a custom `backward(payload)` when the workflow uses conditional
logic or composition rules that auto-backward does not support. Custom backward
implementations receive:

- `payload["structure"]`: the selected structure dictionary.
- `payload["metrics"]`: per-agent metric DataFrames with `setting`,
  `accuracy`, and `latency`.

## 5. Update Runtime Preprocess and Trace Logic If Needed

If the new workflow changes expected inputs or outputs, update the relevant
helpers in `workflow_compiler/dsl/runtime.py`:

- `_preprocess_query`
- `_build_trace_*`

These functions control how raw dataset rows become DSL inputs and how runtime
execution writes trace entries.

## 6. Re-export the Workflow Class

If useful, export the new DSL class from the package `__init__.py` for convenient imports.

## 7. Sanity Check

Validate that:

- the DSL compiles
- a single sample executes successfully
- the trace output contains the expected final fields
- evaluation metadata is attached later by validation rather than being generated inside the workflow
- inferred structures look reasonable:

```bash
python - <<'PY'
from workflow_compiler.workflows.dsl_registry import get_workflow_module

workflow = get_workflow_module("myworkflow")
for structure in workflow.enumerate_structures():
    print(structure["structure_id"], structure["active_agent_counts"])
PY
```

## Related API

The workflow registry helpers are documented in the curated API page for `workflow_compiler.workflows`.
