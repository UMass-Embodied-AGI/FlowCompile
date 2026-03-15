"""Automatic backward synthesis for Python DSL workflows.

This module provides a default `backward(payload)` implementation that:
- composes workflow accuracy from profiled sub-agent accuracies
- composes workflow latency from profiled sub-agent latencies
- supports loop-break retry pattern (`if <tool_field>: break`) used by code workflow

Only sequential latency execution is implemented right now, but the mode
dispatch is intentionally extensible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from workflow_compiler.dsl.structures import apply_structure


DEFAULT_EXECUTION_MODE = "sequential"


@dataclass(frozen=True)
class WorkflowLoopSpec:
    name: str
    count: int
    map_nodes: Tuple[str, ...]
    reduce_node: Optional[str] = None


def _state_ref_parts(obj: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(obj, dict):
        return None
    ref = obj.get("ref")
    if not isinstance(ref, str) or not ref.startswith("state."):
        return None
    parts = ref.split(".")
    if len(parts) < 2:
        return None
    node_id = parts[1] or ""
    if not node_id:
        return None
    field = ".".join(parts[2:]) if len(parts) > 2 else ""
    return node_id, field


def _collect_direct_state_refs(obj: Any) -> List[str]:
    refs: List[str] = []
    parts = _state_ref_parts(obj)
    if parts is not None:
        node_id, field = parts
        if field == "":
            refs.append(node_id)
        return refs
    if isinstance(obj, list):
        for item in obj:
            refs.extend(_collect_direct_state_refs(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            refs.extend(_collect_direct_state_refs(value))
    return refs


def _collect_list_direct_state_ref_groups(obj: Any, groups: List[List[str]]) -> List[str]:
    parts = _state_ref_parts(obj)
    if parts is not None:
        node_id, field = parts
        if field == "":
            return [node_id]
        return []
    if isinstance(obj, list):
        refs: List[str] = []
        for item in obj:
            refs.extend(_collect_list_direct_state_ref_groups(item, groups))
        if refs:
            groups.append(refs)
        return refs
    if isinstance(obj, dict):
        refs: List[str] = []
        for value in obj.values():
            refs.extend(_collect_list_direct_state_ref_groups(value, groups))
        return refs
    return []


def _collect_non_list_direct_state_refs(obj: Any, in_list: bool = False) -> List[str]:
    refs: List[str] = []
    parts = _state_ref_parts(obj)
    if parts is not None:
        node_id, field = parts
        if field == "" and not in_list:
            refs.append(node_id)
        return refs
    if isinstance(obj, list):
        for item in obj:
            refs.extend(_collect_non_list_direct_state_refs(item, in_list=True))
        return refs
    if isinstance(obj, dict):
        for value in obj.values():
            refs.extend(_collect_non_list_direct_state_refs(value, in_list=in_list))
    return refs


def _collect_direct_state_refs_by_key(inputs: Dict[str, Any], key: str) -> List[str]:
    value = inputs.get(key)
    if value is None:
        return []
    return _collect_direct_state_refs(value)


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


@dataclass
class RetryPattern:
    test_node_ids: List[str]
    fixer_agent: Optional[str]
    fixer_node_ids: List[str]
    base_solution_node_id: str


def _condition_matches_test_passed(edge: Dict[str, Any]) -> bool:
    when = edge.get("when")
    if not isinstance(when, dict):
        return False
    if "all" in when or "any" in when:
        return False
    op = when.get("op")
    path = when.get("path")
    if op not in ("truthy", "falsy"):
        return False
    if not isinstance(path, str):
        return False
    src = edge.get("from")
    if not isinstance(src, str) or not src:
        return False
    return path == f"state.{src}.test_passed"


def _detect_retry_pattern(spec: Dict[str, Any], node_order: Dict[str, int]) -> Optional[RetryPattern]:
    nodes = spec.get("nodes", []) or []
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
    conditional_edges = [edge for edge in (spec.get("edges", []) or []) if edge.get("when") is not None]
    if not conditional_edges:
        return None

    for edge in conditional_edges:
        if not _condition_matches_test_passed(edge):
            return None
        src = str(edge.get("from"))
        src_node = node_by_id.get(src)
        if not isinstance(src_node, dict) or src_node.get("type") != "tool":
            return None

    test_node_ids = _unique(str(edge.get("from")) for edge in conditional_edges)
    test_node_ids = sorted(test_node_ids, key=lambda node_id: node_order.get(node_id, -1))
    if not test_node_ids:
        return None

    fixer_candidates: Set[str] = set()
    for edge in conditional_edges:
        when = edge.get("when") or {}
        if when.get("op") != "falsy":
            continue
        dst = str(edge.get("to"))
        dst_node = node_by_id.get(dst)
        if not isinstance(dst_node, dict):
            return None
        if dst_node.get("type") == "agent":
            fixer_name = dst_node.get("name")
            if isinstance(fixer_name, str) and fixer_name:
                fixer_candidates.add(fixer_name)

    if len(fixer_candidates) > 1:
        return None
    fixer_agent = next(iter(fixer_candidates)) if fixer_candidates else None

    first_test_node = node_by_id.get(test_node_ids[0]) or {}
    io = first_test_node.get("io") or {}
    inputs = io.get("inputs") if isinstance(io, dict) else None
    inputs = inputs if isinstance(inputs, dict) else {}
    solution_refs = _collect_direct_state_refs_by_key(inputs, "solution")
    if not solution_refs:
        direct_refs = _collect_direct_state_refs(inputs)
        if not direct_refs:
            return None
        base_solution_node_id = direct_refs[0]
    else:
        base_solution_node_id = solution_refs[0]

    fixer_node_ids: List[str] = []
    if fixer_agent:
        fixer_node_ids = sorted(
            [
                node_id
                for node_id, node in node_by_id.items()
                if node.get("type") == "agent" and node.get("name") == fixer_agent
            ],
            key=lambda node_id: node_order.get(node_id, -1),
        )

    return RetryPattern(
        test_node_ids=test_node_ids,
        fixer_agent=fixer_agent,
        fixer_node_ids=fixer_node_ids,
        base_solution_node_id=base_solution_node_id,
    )


def _preferred_output_node_id(spec: Dict[str, Any]) -> Optional[str]:
    outputs = spec.get("outputs")
    if not isinstance(outputs, dict):
        refs = _collect_direct_state_refs(outputs)
        return refs[0] if refs else None

    for key in ("final_solution", "final_answer", "full_solution"):
        if key not in outputs:
            continue
        refs = _collect_direct_state_refs(outputs.get(key))
        if refs:
            return refs[0]
    refs = _collect_direct_state_refs(outputs)
    return refs[0] if refs else None


def _expected_fix_attempts(p_initial_correct: Any, p_fix_code: Any, max_attempts: int) -> Any:
    if max_attempts <= 0:
        return 0.0

    p_initial = np.asarray(p_initial_correct)
    p_fix = np.asarray(p_fix_code)
    p_need_fix = 1 - p_initial

    r = 1 - p_fix
    epsilon = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1 - r**max_attempts) / np.maximum(1 - r, epsilon)
        expected_when_fixing = np.where(
            p_fix >= 1.0,
            1.0,
            np.where(np.abs(1 - r) < epsilon, float(max_attempts), ratio),
        )
    return p_need_fix * expected_when_fixing


def _collect_candidate_ref_ids_from_full_spec(full_spec: Dict[str, Any]) -> Set[str]:
    candidate_ids: Set[str] = set()
    for node in full_spec.get("nodes", []) or []:
        if node.get("type") != "agent":
            continue
        io = node.get("io") or {}
        inputs = io.get("inputs") if isinstance(io, dict) else None
        if not isinstance(inputs, dict):
            continue
        groups: List[List[str]] = []
        _collect_list_direct_state_ref_groups(inputs, groups)
        for group in groups:
            for ref_id in group:
                candidate_ids.add(ref_id)
    return candidate_ids


def _node_operator(node: Dict[str, Any], full_node: Optional[Dict[str, Any]]) -> str:
    for candidate in (node, full_node):
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            continue
        op = metadata.get("operator")
        if isinstance(op, str) and op.strip():
            return op.strip().lower()
    return "sequential"


def _node_local_success_probability(
    node_id: str,
    node_by_id: Dict[str, Dict[str, Any]],
    ctx: Any,
) -> Any:
    node = node_by_id.get(node_id)
    if not isinstance(node, dict):
        return 0.0
    node_type = node.get("type")
    if node_type == "agent":
        return ctx.acc(str(node.get("name")), 0.0)
    if node_type in {"tool", "end"}:
        return 1.0
    return 1.0


def _node_success_probability(
    node_id: str,
    node_by_id: Dict[str, Dict[str, Any]],
    full_node_by_id: Dict[str, Dict[str, Any]],
    active_node_ids: Set[str],
    ctx: Any,
    memo: Dict[str, Any],
    visiting: Set[str],
) -> Any:
    if node_id in memo:
        return memo[node_id]
    if node_id in visiting:
        raise ValueError(f"Cycle detected while composing accuracy at node '{node_id}'.")

    node = node_by_id.get(node_id)
    if node is None:
        memo[node_id] = 0.0
        return memo[node_id]

    visiting.add(node_id)
    io = node.get("io") or {}
    inputs = io.get("inputs") if isinstance(io, dict) else None
    inputs = inputs if isinstance(inputs, dict) else {}

    groups: List[List[str]] = []
    full_node = full_node_by_id.get(node_id)
    full_io = full_node.get("io") if isinstance(full_node, dict) else None
    full_inputs = full_io.get("inputs") if isinstance(full_io, dict) else None
    if isinstance(full_inputs, dict):
        _collect_list_direct_state_ref_groups(full_inputs, groups)
        groups = [[ref for ref in group if ref in active_node_ids] for group in groups]
        groups = [group for group in groups if group]
    else:
        _collect_list_direct_state_ref_groups(inputs, groups)
    single_refs = _unique(_collect_non_list_direct_state_refs(inputs))

    input_prob: Any = 1.0
    operator = _node_operator(node, full_node)
    for ref in single_refs:
        input_prob = input_prob * _node_success_probability(
            ref, node_by_id, full_node_by_id, active_node_ids, ctx, memo, visiting
        )
    for group in groups:
        refs = _unique(group)
        if operator in {"map_reduce", "map-reduce", "reduce"}:
            # For map-reduce operators, list dependencies compose multiplicatively
            # by stage/operator success, not by ensemble-style OR semantics.
            group_prob: Any = 1.0
            for ref in refs:
                group_prob = group_prob * _node_local_success_probability(ref, node_by_id, ctx)
            input_prob = input_prob * group_prob
        else:
            fail_prob: Any = 1.0
            for ref in refs:
                fail_prob = fail_prob * (
                    1
                    - _node_success_probability(
                        ref, node_by_id, full_node_by_id, active_node_ids, ctx, memo, visiting
                    )
                )
            input_prob = input_prob * (1 - fail_prob)

    node_type = node.get("type")
    if node_type == "agent":
        prob = input_prob * ctx.acc(str(node.get("name")), 0.0)
    elif node_type == "tool":
        prob = input_prob
    elif node_type == "end":
        prob = 1.0
    else:
        prob = input_prob

    visiting.remove(node_id)
    memo[node_id] = prob
    return prob


def _compute_workflow_accuracy(
    spec: Dict[str, Any],
    full_spec: Dict[str, Any],
    ctx: Any,
    retry_pattern: Optional[RetryPattern],
) -> Any:
    nodes = spec.get("nodes", []) or []
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
    full_nodes = full_spec.get("nodes", []) or []
    full_node_by_id = {str(node.get("id")): node for node in full_nodes if node.get("id")}
    active_node_ids = set(node_by_id.keys())

    memo: Dict[str, Any] = {}
    if retry_pattern is not None:
        p_initial = _node_success_probability(
            retry_pattern.base_solution_node_id,
            node_by_id,
            full_node_by_id,
            active_node_ids,
            ctx,
            memo,
            set(),
        )
        if retry_pattern.fixer_agent and retry_pattern.fixer_node_ids:
            k = len(retry_pattern.fixer_node_ids)
            p_fix = ctx.acc(retry_pattern.fixer_agent, 0.0)
            p_fix_success = 1 - (1 - p_fix) ** k
            return p_initial + (1 - p_initial) * p_fix_success
        return p_initial

    output_node_id = _preferred_output_node_id(spec)
    if output_node_id is None:
        return 0.0
    output_prob = _node_success_probability(
        output_node_id,
        node_by_id,
        full_node_by_id,
        active_node_ids,
        ctx,
        memo,
        set(),
    )

    # If the workflow includes list-based candidate branches in the full graph,
    # but none of those candidates are active in this structure, accuracy should
    # be zero (e.g. programmer-only math structure).
    candidate_ids = _collect_candidate_ref_ids_from_full_spec(full_spec)
    if candidate_ids:
        active_node_ids = {str(node.get("id")) for node in nodes if node.get("id")}
        if not any(candidate in active_node_ids for candidate in candidate_ids):
            return 0.0

    return output_prob


def _compose_latency_sequential_with_initial(
    ctx: Any,
    retry_pattern: Optional[RetryPattern],
    p_initial_correct: Any,
) -> Any:
    total: Any = 0.0
    for agent, count in ctx.active_agent_counts.items():
        total = total + ctx.lat(agent, 0.0) * int(count)

    if retry_pattern is None or not retry_pattern.fixer_agent or not retry_pattern.fixer_node_ids:
        return total

    fixer_agent = retry_pattern.fixer_agent
    max_attempts = len(retry_pattern.fixer_node_ids)
    p_fix = ctx.acc(fixer_agent, 0.0)
    expected_attempts = _expected_fix_attempts(
        p_initial_correct=p_initial_correct,
        p_fix_code=p_fix,
        max_attempts=max_attempts,
    )
    total = total + (expected_attempts - max_attempts) * ctx.lat(fixer_agent, 0.0)
    return total


def _parse_workflow_loops(raw: Any) -> List[WorkflowLoopSpec]:
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise ValueError("workflow_loops must be a list of loop definitions.")

    loops: List[WorkflowLoopSpec] = []
    seen_names: Set[str] = set()
    assigned_nodes: Dict[str, str] = {}

    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"workflow_loops[{idx}] must be a mapping.")

        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError(f"workflow_loops[{idx}].name must be a non-empty string.")
        if name in seen_names:
            raise ValueError(f"workflow_loops contains duplicate loop name '{name}'.")
        seen_names.add(name)

        count = item.get("count")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"workflow_loops[{idx}].count must be an integer >= 1.")

        raw_map_nodes = item.get("map_nodes")
        if not isinstance(raw_map_nodes, list) or not raw_map_nodes:
            raise ValueError(f"workflow_loops[{idx}].map_nodes must be a non-empty list.")
        map_nodes: List[str] = []
        seen_local: Set[str] = set()
        for node_idx, value in enumerate(raw_map_nodes):
            node_id = str(value or "").strip()
            if not node_id:
                raise ValueError(
                    f"workflow_loops[{idx}].map_nodes[{node_idx}] must be a non-empty string."
                )
            if node_id in seen_local:
                raise ValueError(
                    f"workflow_loops[{idx}] contains duplicate node '{node_id}' in map_nodes."
                )
            seen_local.add(node_id)
            owner = assigned_nodes.get(node_id)
            if owner is not None:
                raise ValueError(
                    f"workflow_loops node '{node_id}' is assigned to both '{owner}' and '{name}'."
                )
            assigned_nodes[node_id] = name
            map_nodes.append(node_id)

        reduce_raw = item.get("reduce_node")
        reduce_node: Optional[str] = None
        if reduce_raw is not None:
            reduce_node = str(reduce_raw or "").strip()
            if not reduce_node:
                raise ValueError(f"workflow_loops[{idx}].reduce_node must be a non-empty string.")
            if reduce_node in seen_local:
                raise ValueError(
                    f"workflow_loops[{idx}].reduce_node '{reduce_node}' cannot also appear in map_nodes."
                )
            owner = assigned_nodes.get(reduce_node)
            if owner is not None:
                raise ValueError(
                    f"workflow_loops node '{reduce_node}' is assigned to both '{owner}' and '{name}'."
                )
            assigned_nodes[reduce_node] = name

        loops.append(
            WorkflowLoopSpec(
                name=name,
                count=count,
                map_nodes=tuple(map_nodes),
                reduce_node=reduce_node,
            )
        )

    return loops


def _compose_latency_with_workflow_loops(
    spec: Dict[str, Any],
    ctx: Any,
    retry_pattern: Optional[RetryPattern],
    p_initial_correct: Any,
    default_composer: "LatencyComposer",
) -> Any:
    loops = _parse_workflow_loops((getattr(ctx, "metadata", None) or {}).get("workflow_loops"))
    if not loops:
        return default_composer(ctx, retry_pattern, p_initial_correct)

    nodes = spec.get("nodes", []) or []
    active_agent_nodes = [
        node
        for node in nodes
        if node.get("type") == "agent" and node.get("id") and ctx.enabled(str(node.get("name")))
    ]
    node_by_id = {str(node.get("id")): node for node in active_agent_nodes}
    multipliers: Dict[str, int] = {node_id: 1 for node_id in node_by_id}

    for loop in loops:
        for node_id in loop.map_nodes:
            node = node_by_id.get(node_id)
            if node is None:
                raise ValueError(
                    f"workflow_loops loop '{loop.name}' references unknown or inactive map node '{node_id}'."
                )
            multipliers[node_id] = loop.count

        if loop.reduce_node is not None:
            node = node_by_id.get(loop.reduce_node)
            if node is None:
                raise ValueError(
                    f"workflow_loops loop '{loop.name}' references unknown or inactive reduce node '{loop.reduce_node}'."
                )
            operator = _node_operator(node, node)
            if operator not in {"map_reduce", "map-reduce", "reduce"}:
                raise ValueError(
                    f"workflow_loops loop '{loop.name}' reduce node '{loop.reduce_node}' "
                    f"must use operator map_reduce or reduce, found '{operator}'."
                )
            multipliers[loop.reduce_node] = 1

    total: Any = 0.0
    for node in active_agent_nodes:
        node_id = str(node.get("id"))
        agent = str(node.get("name"))
        total = total + ctx.lat(agent, 0.0) * multipliers[node_id]

    if retry_pattern is None or not retry_pattern.fixer_agent or not retry_pattern.fixer_node_ids:
        return total

    fixer_agent = retry_pattern.fixer_agent
    max_attempts = len(retry_pattern.fixer_node_ids)
    p_fix = ctx.acc(fixer_agent, 0.0)
    expected_attempts = _expected_fix_attempts(
        p_initial_correct=p_initial_correct,
        p_fix_code=p_fix,
        max_attempts=max_attempts,
    )
    extra_attempts = expected_attempts - max_attempts
    if np.any(np.asarray(extra_attempts) != 0):
        total = total + extra_attempts * ctx.lat(fixer_agent, 0.0)
    return total


LatencyComposer = Callable[[Any, Optional[RetryPattern], Any], Any]

_LATENCY_COMPOSERS: Dict[str, LatencyComposer] = {
    "sequential": _compose_latency_sequential_with_initial,
}


def supported_execution_modes() -> Tuple[str, ...]:
    return tuple(sorted(_LATENCY_COMPOSERS.keys()))


def validate_execution_mode(execution_mode: str) -> str:
    mode = str(execution_mode or "").strip().lower()
    if mode in _LATENCY_COMPOSERS:
        return mode
    supported = ", ".join(supported_execution_modes())
    raise ValueError(
        f"Unsupported execution_mode '{execution_mode}'. Supported modes: {supported}"
    )


def auto_backward(workflow: Any, payload: Dict[str, Any]) -> pd.DataFrame:
    ctx = workflow.metric_context(payload)
    full_spec = workflow._compile_cached()
    workflow_type = getattr(workflow, "workflow_type", "unknown")
    spec = apply_structure(full_spec, ctx.structure, workflow_type)

    node_order = {
        str(node.get("id")): idx
        for idx, node in enumerate(spec.get("nodes", []) or [])
        if node.get("id")
    }
    conditional_edges = [edge for edge in (spec.get("edges", []) or []) if edge.get("when") is not None]
    retry_pattern = _detect_retry_pattern(spec, node_order)
    if conditional_edges and retry_pattern is None:
        raise ValueError(
            "Auto backward currently supports only captured loop-break conditionals "
            "(`if <cond>: break`) backed by tool `test_passed` checks. "
            "Provide a manual backward() implementation for this workflow."
        )

    p_initial_for_latency: Any = None
    if retry_pattern is not None:
        nodes = spec.get("nodes", []) or []
        node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
        full_nodes = full_spec.get("nodes", []) or []
        full_node_by_id = {str(node.get("id")): node for node in full_nodes if node.get("id")}
        active_node_ids = set(node_by_id.keys())
        p_initial_for_latency = _node_success_probability(
            retry_pattern.base_solution_node_id,
            node_by_id,
            full_node_by_id,
            active_node_ids,
            ctx,
            memo={},
            visiting=set(),
        )

    workflow_accuracy = _compute_workflow_accuracy(spec, full_spec, ctx, retry_pattern)

    mode = validate_execution_mode(getattr(workflow, "execution_mode", DEFAULT_EXECUTION_MODE))
    composer = _LATENCY_COMPOSERS[mode]
    workflow_latency = _compose_latency_with_workflow_loops(
        spec,
        ctx,
        retry_pattern,
        p_initial_for_latency,
        composer,
    )

    return ctx.finish(
        workflow_accuracy=workflow_accuracy,
        workflow_latency=workflow_latency,
    )


__all__ = [
    "DEFAULT_EXECUTION_MODE",
    "supported_execution_modes",
    "validate_execution_mode",
    "auto_backward",
]
