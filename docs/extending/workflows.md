# Adding a Workflow

This guide documents the Python DSL workflow extension path used by FlowCompile. It is adapted from the maintainer notes in `workflow_compiler/workflows/ADDING_WORKFLOW.md`.

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

## 2. Implement the DSL Workflow

A typical DSL workflow subclasses `WorkflowModule` and defines a `forward(query)` method:

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

- Use standard Python control flow in the DSL.
- Do not embed evaluation logic in the workflow definition.
- Keep the input surface centered on a single `query` object.
- Keep outputs aligned with the runtime trace builders:
  - `final_answer`
  - `full_solution`
  - `final_solution`
- Prefer inferred structure enumeration rather than hand-maintained enumerators.
- Keep `execution_mode="sequential"` unless the runtime adds explicit support for more modes.

## 3. Register the Workflow

Update `workflow_compiler/workflows/dsl_registry.py` so the new `workflow_type` resolves to your DSL class.

## 4. Update Runtime Preprocess and Trace Logic If Needed

If the new workflow changes expected inputs or outputs, update the relevant helpers in `workflow_compiler/dsl/runtime.py`:

- `_preprocess_query`
- `_build_trace_*`

## 5. Re-export the Workflow Class

If useful, export the new DSL class from the package `__init__.py` for convenient imports.

## 6. Sanity Check

Validate that:

- the DSL compiles
- a single sample executes successfully
- the trace output contains the expected final fields
- evaluation metadata is attached later by validation rather than being generated inside the workflow

## Related API

The workflow registry helpers are documented in the curated API page for `workflow_compiler.workflows`.

