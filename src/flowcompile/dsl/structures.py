"""Structure parsing and pruning utilities for DSL workflows."""
from __future__ import annotations

import copy
from typing import Dict, Any, List, Optional, Tuple, Union


def _remove_nodes(spec: Dict[str, Any], remove_ids: set) -> Dict[str, Any]:
    spec = copy.deepcopy(spec)
    if not remove_ids:
        return spec

    nodes = [n for n in spec.get("nodes", []) if n.get("id") not in remove_ids]
    node_order = [n.get("id") for n in spec.get("nodes", [])]
    node_pos = {nid: idx for idx, nid in enumerate(node_order)}

    # Filter and bypass edges
    edges = spec.get("edges", [])
    incoming: Dict[str, List[Dict[str, Any]]] = {}
    outgoing: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        incoming.setdefault(e.get("to"), []).append(e)
        outgoing.setdefault(e.get("from"), []).append(e)

    # Determine replacement for removed nodes by following graph predecessors.
    # This preserves "pass input through pruned node" semantics.
    replacement: Dict[str, str] = {}

    def _resolve_replacement(node_id: str, visiting: set) -> Any:
        if node_id not in remove_ids:
            return node_id
        if node_id in replacement:
            return replacement[node_id]
        if node_id in visiting:
            return None

        visiting.add(node_id)
        candidates: List[str] = []
        for inc in incoming.get(node_id, []):
            src = inc.get("from")
            if not src:
                continue
            resolved = _resolve_replacement(src, visiting)
            if resolved and resolved not in remove_ids:
                candidates.append(resolved)
        visiting.remove(node_id)

        if candidates:
            # Prefer the candidate closest to the removed node in original order.
            candidates = sorted(set(candidates), key=lambda nid: node_pos.get(nid, -1))
            replacement[node_id] = candidates[-1]
            return replacement[node_id]

        # Fallback: nearest previous non-removed node in declaration order.
        idx = node_pos.get(node_id, -1)
        if idx >= 0:
            for j in range(idx - 1, -1, -1):
                cand = node_order[j]
                if cand not in remove_ids:
                    replacement[node_id] = cand
                    return cand
        return None

    for rid in remove_ids:
        _resolve_replacement(rid, set())

    new_edges: List[Dict[str, Any]] = []
    edge_keys = set()

    def _add_edge(src: str, dst: str, when: Any = None) -> None:
        key = (src, dst, repr(when))
        if key in edge_keys:
            return
        edge_keys.add(key)
        edge: Dict[str, Any] = {"from": src, "to": dst}
        if when is not None:
            edge["when"] = when
        new_edges.append(edge)

    for e in edges:
        if e.get("from") in remove_ids or e.get("to") in remove_ids:
            continue
        _add_edge(e.get("from"), e.get("to"), e.get("when"))

    # Bypass removed-node chains.
    # For each non-removed source, traverse through removed nodes until we hit
    # non-removed destinations, then connect source -> destination directly.
    non_removed_ids = {n.get("id") for n in nodes}
    for src in non_removed_ids:
        for out in outgoing.get(src, []):
            first = out.get("to")
            if first not in remove_ids:
                continue

            stack: List[Tuple[str, Any]] = [(first, out.get("when"))]
            seen = set()
            while stack:
                current, inherited_when = stack.pop()
                visit_key = (current, repr(inherited_when))
                if visit_key in seen:
                    continue
                seen.add(visit_key)

                for next_edge in outgoing.get(current, []):
                    nxt = next_edge.get("to")
                    next_when = inherited_when if inherited_when is not None else next_edge.get("when")
                    if nxt in remove_ids:
                        stack.append((nxt, next_when))
                    elif nxt in non_removed_ids:
                        _add_edge(src, nxt, next_when)

    # Update entry
    entry = spec.get("entry")
    if entry in remove_ids:
        for nid in node_order:
            if nid not in remove_ids:
                entry = nid
                break

    # Rewrite refs in node inputs and outputs
    def rewrite_ref(obj: Any, in_list: bool = False) -> Any:
        if isinstance(obj, dict) and "ref" in obj:
            ref = obj.get("ref")
            if ref and ref.startswith("state."):
                node_id = ref.split(".")[1]
                if node_id in remove_ids:
                    if in_list:
                        return None
                    if node_id in replacement:
                        return {"ref": ref.replace(f"state.{node_id}", f"state.{replacement[node_id]}")}
                    return obj  # keep as-is
                if node_id in replacement:
                    return {"ref": ref.replace(f"state.{node_id}", f"state.{replacement[node_id]}")}
            return obj
        if isinstance(obj, list):
            out = []
            seen_refs = set()
            for v in obj:
                rv = rewrite_ref(v, in_list=True)
                if rv is None:
                    continue
                if isinstance(rv, dict) and "ref" in rv:
                    ref = rv.get("ref")
                    if isinstance(ref, str) and ref in seen_refs:
                        continue
                    if isinstance(ref, str):
                        seen_refs.add(ref)
                out.append(rv)
            return out
        if isinstance(obj, dict):
            return {k: rewrite_ref(v, in_list=False) for k, v in obj.items()}
        return obj

    for node in nodes:
        io = node.get("io")
        if isinstance(io, dict):
            node["io"] = rewrite_ref(io)

    outputs = spec.get("outputs")
    if outputs is not None:
        outputs = rewrite_ref(outputs)

    return {
        **spec,
        "nodes": nodes,
        "edges": new_edges,
        "entry": entry,
        "outputs": outputs,
    }


def apply_structure(
    spec: Dict[str, Any],
    structure: Optional[Union[str, Dict[str, Any]]],
    workflow_type: str,
) -> Dict[str, Any]:
    if not structure:
        return copy.deepcopy(spec)

    if isinstance(structure, dict):
        structure_obj = structure
    else:
        from flowcompile.workflows.dsl_registry import get_workflow_module

        workflow_module = get_workflow_module(workflow_type)
        structure_obj = workflow_module.get_structure(str(structure))

    remove_ids = set(structure_obj.get("remove_node_ids") or [])
    return _remove_nodes(spec, remove_ids)
