"""Runtime integration for Python DSL workflows."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import copy

from workflow_compiler.dsl.executor import DslExecutor
from workflow_compiler.dsl.structures import apply_structure
from workflow_compiler.workflows.dsl_registry import get_workflow_module
from workflow_compiler.core.logs import logger


def _strip_difficulty_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        return sample
    cleaned = copy.deepcopy(sample)
    cleaned.pop("level", None)
    cleaned.pop("difficulty", None)
    metadata = cleaned.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("level", None)
        metadata.pop("difficulty", None)
        original_sample = metadata.get("original_sample")
        if isinstance(original_sample, dict):
            original_sample.pop("level", None)
            original_sample.pop("difficulty", None)
    return cleaned


def _build_query_payload(problem: str, entry_point: str = "", question_id: str = "") -> Dict[str, str]:
    return {
        "problem": problem,
        "entry_point": entry_point or "",
        "question_id": str(question_id or ""),
    }


def _preprocess_query(workflow_type: str, query: Dict[str, Any], config: Dict[str, Any]):
    workflow_type = workflow_type.lower()
    if not isinstance(query, dict):
        raise TypeError("Workflow query must be a dict payload.")
    sanitized_query = _strip_difficulty_fields(query)

    if workflow_type in ("math", "gsm8k"):
        problem_text = query.get("problem", str(query))
        question_id = query.get("question_id", query.get("_id", query.get("id", "")))
        return {
            "inputs": {"query": _build_query_payload(problem=problem_text, question_id=question_id)},
            "problem_text": problem_text,
            "original_sample": sanitized_query,
        }

    if workflow_type == "hotpotqa":
        question = query.get("question", "")
        context_items = query.get("context", [])
        paragraphs = [item[1] for item in context_items if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(p) for p in paragraphs)
        problem_text = f"Context: {context_str}\n\nQuestion: {question}\n\nAnswer:"
        question_id = query.get("question_id", query.get("_id", query.get("id", "")))
        return {
            "inputs": {"query": _build_query_payload(problem=problem_text, question_id=question_id)},
            "problem_text": problem_text,
            "original_sample": sanitized_query,
        }

    if workflow_type == "livecodebench":
        problem_text = query.get("question", query.get("text", query.get("problem", str(query))))
        entry_point = query.get("entry_point", query.get("function_name", ""))
        question_id = query.get("question_id", query.get("id", ""))
        starter_code = query.get("canonical_solution", "")
        if starter_code:
            problem_text = f"{problem_text}\n\nStarter Code:\n{starter_code}"
        return {
            "inputs": {"query": _build_query_payload(problem=problem_text, entry_point=entry_point, question_id=question_id)},
            "problem_text": problem_text,
            "original_sample": sanitized_query,
            "question_id": str(question_id),
        }

    question_id = query.get("question_id", query.get("_id", query.get("id", "")))
    return {
        "inputs": {"query": _build_query_payload(problem=str(query), question_id=question_id)},
        "problem_text": str(query),
        "original_sample": sanitized_query,
    }


def _structure_trace_fields(structure: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(structure, dict):
        return {}
    return {
        "structure_id": structure.get("structure_id"),
        "total_branches": structure.get("total_branches"),
        "is_full": bool(structure.get("is_full", False)),
        "active_agent_counts": structure.get("active_agent_counts", {}),
    }


def _build_trace_math(steps, outputs, structure, preprocess):
    problem_text = preprocess.get("problem_text")
    original_sample = preprocess.get("original_sample")
    structure_fields = _structure_trace_fields(structure)

    final_answer = outputs.get("final_answer")
    full_solution = outputs.get("full_solution", final_answer)

    trace_entry = {
        "timestamp": datetime.now().isoformat(),
        "problem": problem_text,
        "final_answer": final_answer,
        "full_solution": full_solution,
        "workflow_type": "dsl_math",
        "structure_id": structure_fields.get("structure_id"),
        "num_solutions": sum(1 for s in steps if s.get("agent") in ("generate_solver", "detailed_solver", "refine_solver")),
        "steps": steps,
        "num_steps": len(steps),
        "metadata": {
            "original_sample": original_sample,
            "structure": structure_fields,
        },
    }
    return trace_entry


def _build_trace_hotpotqa(steps, outputs, structure, preprocess):
    problem_text = preprocess.get("problem_text")
    original_sample = preprocess.get("original_sample")
    structure_fields = _structure_trace_fields(structure)

    final_answer = outputs.get("final_answer")

    trace_entry = {
        "timestamp": datetime.now().isoformat(),
        "problem": problem_text,
        "final_answer": final_answer,
        "full_solution": final_answer,
        "workflow_type": "dsl_hotpotqa",
        "structure_id": structure_fields.get("structure_id"),
        "steps": steps,
        "num_steps": len(steps),
        "metadata": {
            "original_sample": original_sample,
            "structure": structure_fields,
        },
    }
    return trace_entry


def _build_trace_livecodebench(steps, outputs, structure, preprocess):
    problem_text = preprocess.get("problem_text")
    original_sample = preprocess.get("original_sample")
    question_id = preprocess.get("question_id", "")
    structure_fields = _structure_trace_fields(structure)

    final_solution = outputs.get("full_solution", outputs.get("final_solution", outputs.get("final_answer")))

    filtered_sample = None
    if original_sample:
        try:
            filtered_sample = {
                "question_id": original_sample.get("question_id"),
                "platform": original_sample.get("metadata", {}).get("platform"),
            }
        except Exception:
            filtered_sample = original_sample

    trace_entry = {
        "timestamp": datetime.now().isoformat(),
        "problem": problem_text,
        "entry_point": preprocess.get("inputs", {}).get("query", {}).get("entry_point"),
        "question_id": question_id,
        "final_answer": final_solution,
        "full_solution": final_solution,
        "final_solution": final_solution,
        "workflow_type": "dsl_livecodebench",
        "structure_id": structure_fields.get("structure_id"),
        "num_solutions": sum(1 for s in steps if s.get("agent") == "code_generate"),
        "steps": steps,
        "num_steps": len(steps),
        "metadata": {
            "original_sample": filtered_sample,
            "dataset": "LiveCodeBench",
            "structure": structure_fields,
        },
    }

    return trace_entry


def build_dsl_config(llm_configs: Dict[str, Any], structure_id: Optional[str] = None) -> Dict[str, Any]:
    agents: Dict[str, Any] = {}
    for name, setting in (llm_configs or {}).items():
        if setting is None:
            continue
        if isinstance(setting, dict):
            agents[name] = setting
        else:
            agents[name] = {"setting": setting}
    config: Dict[str, Any] = {"agents": agents}
    if structure_id:
        config["structure_id"] = structure_id
    return config


async def run_dsl_query(
    query: Dict[str, Any],
    config: Dict[str, Any],
    workflow_type: str,
    output_dir: Path,
    compiled_spec: Optional[Dict[str, Any]] = None,
):
    workflow_module = get_workflow_module(workflow_type)
    spec = copy.deepcopy(compiled_spec) if compiled_spec is not None else workflow_module.compile()
    structure_id = config.get("structure_id")
    if structure_id:
        structure = workflow_module.get_structure(structure_id)
    else:
        structure = workflow_module.get_full_structure()
    spec = apply_structure(spec, structure, workflow_type)

    preprocess = _preprocess_query(workflow_type, query, config)
    executor = DslExecutor(spec, workflow_type, config)

    outputs, steps, _state = await executor.run(preprocess.get("inputs", {}))

    # Build trace entry
    if workflow_type in ("math", "gsm8k"):
        trace_entry = _build_trace_math(steps, outputs, structure, preprocess)
        output_value = outputs.get("final_answer")
    elif workflow_type == "hotpotqa":
        trace_entry = _build_trace_hotpotqa(steps, outputs, structure, preprocess)
        output_value = outputs.get("final_answer")
    elif workflow_type == "livecodebench":
        trace_entry = _build_trace_livecodebench(steps, outputs, structure, preprocess)
        output_value = outputs.get("final_answer", outputs.get("full_solution", outputs.get("final_solution")))
    else:
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "steps": steps,
        }
        output_value = outputs

    # Write trace
    trace_file = output_dir / "trace.jsonl"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_file, "a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")

    return output_value


class DslWorkflowRunner:
    """Workflow runner wrapper that matches the fixed workflow interface."""

    def __init__(
        self,
        name: str,
        llm_configs: Dict[str, Any],
        workflow_type: str,
        output_dir: Path,
        structure_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self.workflow_type = workflow_type
        self.output_dir = output_dir
        self.structure_id = structure_id
        self.trace_file = self.output_dir / "trace.jsonl"
        self._config = build_dsl_config(llm_configs, structure_id)
        self._compiled_spec = get_workflow_module(workflow_type).compile()

    async def __call__(self, query: Any):
        if not isinstance(query, dict):
            raise TypeError("DslWorkflowRunner query must be a dict payload.")
        payload = dict(query)

        return await run_dsl_query(
            query=payload,
            config=self._config,
            workflow_type=self.workflow_type,
            output_dir=self.output_dir,
            compiled_spec=self._compiled_spec,
        )
