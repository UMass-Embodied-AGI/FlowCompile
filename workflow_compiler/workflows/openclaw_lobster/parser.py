"""Lobster YAML -> FlowCompile workflow spec parser."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


_STEP_REF_RE = re.compile(r"\$([A-Za-z0-9_-]+)\.stdout")
_LLM_HINT_RE = re.compile(r"(?:_llm_|llm-task|outlook_llm_)", re.IGNORECASE)
_REDUCE_HINT_RE = re.compile(r"(?:overview|reduce|aggregate|merge|consolidate)", re.IGNORECASE)
_LLM_SCRIPT_RE = re.compile(r"outlook_llm_([A-Za-z0-9_]+)", re.IGNORECASE)


def _extract_step_refs(text: Any) -> Set[str]:
    if not isinstance(text, str) or not text:
        return set()
    return set(_STEP_REF_RE.findall(text))


def _is_llm_step(step: Dict[str, Any]) -> bool:
    if not isinstance(step, dict):
        return False
    step_id = str(step.get("id") or "")
    command = str(step.get("command") or "")
    return bool(_LLM_HINT_RE.search(step_id) or _LLM_HINT_RE.search(command))


def _is_reduce_like(step_id: str, step: Dict[str, Any]) -> bool:
    if _REDUCE_HINT_RE.search(step_id):
        return True
    command = str(step.get("command") or "")
    script_names = _LLM_SCRIPT_RE.findall(command)
    return any(_REDUCE_HINT_RE.search(name) for name in script_names)


def parse_lobster_workflow(path: str) -> Dict[str, Any]:
    workflow_path = Path(path)
    if not workflow_path.exists():
        raise FileNotFoundError(f"Lobster workflow file not found: {workflow_path}")

    payload = yaml.safe_load(workflow_path.read_text(encoding="utf-8")) or {}
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Invalid Lobster workflow (missing non-empty steps): {workflow_path}")

    ordered_step_ids: List[str] = []
    step_by_id: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "").strip()
        if not step_id:
            continue
        ordered_step_ids.append(step_id)
        step_by_id[step_id] = step

    if not ordered_step_ids:
        raise ValueError(f"No valid step IDs found in Lobster workflow: {workflow_path}")

    deps: Dict[str, List[str]] = {}
    for step_id in ordered_step_ids:
        step = step_by_id[step_id]
        references = set()
        references.update(_extract_step_refs(step.get("stdin")))
        references.update(_extract_step_refs(step.get("command")))
        deps[step_id] = sorted(ref for ref in references if ref in step_by_id)

    llm_step_ids = [step_id for step_id in ordered_step_ids if _is_llm_step(step_by_id[step_id])]
    if not llm_step_ids:
        raise ValueError(
            f"No LLM-like steps detected in Lobster workflow: {workflow_path}. "
            "Expected step IDs/commands containing llm markers."
        )
    llm_set = set(llm_step_ids)
    llm_pos = {step_id: idx for idx, step_id in enumerate(llm_step_ids)}

    upstream_cache: Dict[str, List[str]] = {}

    def upstream_llm(step_id: str) -> List[str]:
        cached = upstream_cache.get(step_id)
        if cached is not None:
            return cached
        stack = list(deps.get(step_id, []))
        seen: Set[str] = set()
        found: Set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in llm_set:
                found.add(cur)
                continue
            stack.extend(deps.get(cur, []))
        ordered = sorted(found, key=lambda item: llm_pos[item])
        upstream_cache[step_id] = ordered
        return ordered

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_edges: Set[tuple] = set()

    for step_id in llm_step_ids:
        step = step_by_id[step_id]
        upstream = upstream_llm(step_id)
        reduce_like = len(upstream) >= 2 and _is_reduce_like(step_id, step)
        if reduce_like:
            operator = "map_reduce"
            inputs: Dict[str, Any] = {
                "items": [{"ref": f"state.{upstream_step}"} for upstream_step in upstream]
            }
            edge_sources = upstream
        else:
            operator = "map"
            chosen = upstream[-1:]  # Prefer the latest upstream LLM stage when multiple contexts are available.
            if chosen:
                inputs = {"source": {"ref": f"state.{chosen[0]}"}}
            else:
                inputs = {}
            edge_sources = chosen

        nodes.append(
            {
                "id": step_id,
                "type": "agent",
                "name": step_id,
                "llm_ref": step_id,
                "io": {"inputs": inputs},
                "metadata": {"operator": operator},
            }
        )

        for src in edge_sources:
            key = (src, step_id, operator)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"from": src, "to": step_id, "operator": operator})

    outputs_ref = llm_step_ids[-1]
    spec = {
        "version": "v1",
        "name": str(payload.get("name") or workflow_path.stem),
        "metadata": {
            "source": "lobster_yaml",
            "workflow_type": "openclaw_lobster",
            "workflow_file": str(workflow_path),
        },
        "nodes": nodes,
        "edges": edges,
        "entry": llm_step_ids[0],
        "outputs": {
            "final_answer": {"ref": f"state.{outputs_ref}"},
            "full_solution": {"ref": f"state.{outputs_ref}"},
            "final_solution": {"ref": f"state.{outputs_ref}"},
        },
    }
    return spec


__all__ = ["parse_lobster_workflow"]
