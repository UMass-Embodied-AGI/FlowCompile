"""Automatic structure inference for Python DSL workflows."""
from __future__ import annotations

from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from workflow_compiler.dsl.structures import _remove_nodes


def _state_ref_node_id(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    ref = obj.get("ref")
    if not isinstance(ref, str) or not ref.startswith("state."):
        return None
    parts = ref.split(".")
    if len(parts) < 2:
        return None
    node_id = parts[1]
    return node_id or None


def _collect_state_refs(obj: Any) -> List[str]:
    refs: List[str] = []
    node_id = _state_ref_node_id(obj)
    if node_id is not None:
        refs.append(node_id)
        return refs
    if isinstance(obj, list):
        for item in obj:
            refs.extend(_collect_state_refs(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            refs.extend(_collect_state_refs(value))
    return refs


def _collect_list_state_ref_groups(obj: Any, groups: List[List[str]]) -> List[str]:
    node_id = _state_ref_node_id(obj)
    if node_id is not None:
        return [node_id]
    if isinstance(obj, list):
        refs: List[str] = []
        for item in obj:
            refs.extend(_collect_list_state_ref_groups(item, groups))
        if refs:
            groups.append(refs)
        return refs
    if isinstance(obj, dict):
        refs: List[str] = []
        for value in obj.values():
            refs.extend(_collect_list_state_ref_groups(value, groups))
        return refs
    return []


def _node_min_input_branches(node: Dict[str, Any]) -> int:
    metadata = node.get("metadata")
    if not isinstance(metadata, dict):
        return 1
    raw_value = metadata.get("min_input_branches", 1)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def _spec_is_valid(spec: Dict[str, Any]) -> bool:
    nodes = spec.get("nodes", []) or []
    node_ids = {n.get("id") for n in nodes if n.get("id")}
    if not node_ids:
        return False

    entry = spec.get("entry")
    if not entry or entry not in node_ids:
        return False

    for edge in spec.get("edges", []) or []:
        src = edge.get("from")
        dst = edge.get("to")
        if src not in node_ids or dst not in node_ids:
            return False

    def _validate_refs(obj: Any) -> bool:
        for node_id in _collect_state_refs(obj):
            if node_id not in node_ids:
                return False
        return True

    if not _validate_refs(spec.get("outputs")):
        return False
    for node in nodes:
        io = node.get("io")
        if isinstance(io, dict) and not _validate_refs(io):
            return False

    outgoing: Dict[str, List[str]] = {}
    for edge in spec.get("edges", []) or []:
        outgoing.setdefault(edge.get("from"), []).append(edge.get("to"))

    reachable: Set[str] = set()
    stack = [entry]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        for nxt in outgoing.get(current, []):
            if nxt not in reachable:
                stack.append(nxt)

    for node_id in _collect_state_refs(spec.get("outputs")):
        if node_id not in reachable:
            return False

    return True


def _estimate_total_branches(
    spec: Dict[str, Any],
    active_agent_ids: Set[str],
    all_agent_ids: Set[str],
) -> int:
    largest_group = 0
    for node in spec.get("nodes", []) or []:
        node_id = node.get("id")
        if node_id not in active_agent_ids:
            continue
        io = node.get("io") or {}
        inputs = io.get("inputs") if isinstance(io, dict) else None
        if not isinstance(inputs, dict):
            continue
        groups: List[List[str]] = []
        _collect_list_state_ref_groups(inputs, groups)
        for group in groups:
            active_count = len({ref for ref in group if ref in active_agent_ids and ref in all_agent_ids})
            if active_count > largest_group:
                largest_group = active_count

    if largest_group > 0:
        return int(largest_group)

    outgoing_to_agents: Dict[str, Set[str]] = {node_id: set() for node_id in active_agent_ids}
    for edge in spec.get("edges", []) or []:
        src = edge.get("from")
        dst = edge.get("to")
        if src in active_agent_ids and dst in active_agent_ids:
            outgoing_to_agents.setdefault(src, set()).add(dst)

    leaf_count = sum(1 for node_id in active_agent_ids if len(outgoing_to_agents.get(node_id, set())) == 0)
    return int(max(1, leaf_count))


def _format_structure_id(agent_order: Sequence[str], counts: Dict[str, int]) -> str:
    parts = [f"{agent}-c{int(counts.get(agent, 0))}" for agent in agent_order]
    return "s__" + "__".join(parts)


def infer_structures(
    spec: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    del metadata  # Reserved for future constraints.

    node_order = [node.get("id") for node in spec.get("nodes", []) or [] if node.get("id")]
    node_pos = {node_id: idx for idx, node_id in enumerate(node_order)}
    node_by_id: Dict[str, Dict[str, Any]] = {
        node.get("id"): node for node in spec.get("nodes", []) or [] if node.get("id")
    }

    agent_nodes_in_order: List[Dict[str, Any]] = [
        node for node in spec.get("nodes", []) or [] if node.get("type") == "agent" and node.get("id")
    ]
    all_agent_ids: Set[str] = {node.get("id") for node in agent_nodes_in_order}

    if not agent_nodes_in_order:
        return []

    agent_to_ids: "OrderedDict[str, List[str]]" = OrderedDict()
    for node in agent_nodes_in_order:
        canonical = str(node.get("name") or "")
        if canonical not in agent_to_ids:
            agent_to_ids[canonical] = []
        agent_to_ids[canonical].append(str(node.get("id")))

    ordered_agents = list(agent_to_ids.keys())
    ranges = [range(len(agent_to_ids[agent]) + 1) for agent in ordered_agents]

    dedup: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for choice in product(*ranges):
        counts = {agent: int(choice[idx]) for idx, agent in enumerate(ordered_agents)}

        active_agent_ids: Set[str] = set()
        for agent in ordered_agents:
            active_agent_ids.update(agent_to_ids[agent][: counts[agent]])
        remove_ids = all_agent_ids - active_agent_ids

        # If a removed list-consumer had multiple surviving producer refs,
        # pruning would silently pick one producer via rewiring; treat it as invalid.
        invalid_removed_consumer = False
        for node in agent_nodes_in_order:
            node_id = node.get("id")
            if node_id not in remove_ids:
                continue
            io = node.get("io") or {}
            inputs = io.get("inputs") if isinstance(io, dict) else None
            if not isinstance(inputs, dict):
                continue
            groups: List[List[str]] = []
            _collect_list_state_ref_groups(inputs, groups)
            for group in groups:
                active_refs = {ref for ref in group if ref in active_agent_ids and ref in all_agent_ids}
                if len(active_refs) > 1:
                    invalid_removed_consumer = True
                    break
            if invalid_removed_consumer:
                break
        if invalid_removed_consumer:
            continue

        # Active list-consumers must keep at least one producer ref.
        invalid_active_consumer = False
        for node in agent_nodes_in_order:
            node_id = node.get("id")
            if node_id not in active_agent_ids:
                continue
            io = node.get("io") or {}
            inputs = io.get("inputs") if isinstance(io, dict) else None
            if not isinstance(inputs, dict):
                continue
            groups: List[List[str]] = []
            _collect_list_state_ref_groups(inputs, groups)
            required_branches = _node_min_input_branches(node)
            for group in groups:
                agent_refs = {ref for ref in group if ref in all_agent_ids}
                if not agent_refs:
                    continue
                active_refs = {ref for ref in agent_refs if ref in active_agent_ids}
                if not active_refs:
                    invalid_active_consumer = True
                    break
                if len(active_refs) < required_branches:
                    invalid_active_consumer = True
                    break
            if invalid_active_consumer:
                break
        if invalid_active_consumer:
            continue

        remove_ids_tuple = tuple(sorted(remove_ids, key=lambda node_id: node_pos.get(node_id, -1)))
        pruned = _remove_nodes(spec, set(remove_ids_tuple))
        if not _spec_is_valid(pruned):
            continue

        total_branches = _estimate_total_branches(pruned, active_agent_ids, all_agent_ids)
        structure = {
            "structure_id": _format_structure_id(ordered_agents, counts),
            "active_agent_counts": {agent: int(counts.get(agent, 0)) for agent in ordered_agents},
            "active_node_ids": [node_id for node_id in node_order if node_id in active_agent_ids],
            "remove_node_ids": list(remove_ids_tuple),
            "total_branches": int(total_branches),
            "is_full": len(remove_ids_tuple) == 0,
        }
        dedup.setdefault(remove_ids_tuple, structure)

    structures = sorted(dedup.values(), key=lambda item: item.get("structure_id", ""))
    return structures


__all__ = [
    "infer_structures",
]
