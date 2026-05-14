from __future__ import annotations

from typing import Any, List

from flowcompile.workflows.dsl_registry import get_workflow_module
from flowcompile.dsl.structures import apply_structure


def _collect_state_refs(obj: Any) -> List[str]:
    refs: List[str] = []
    if isinstance(obj, dict):
        if "ref" in obj:
            ref = obj.get("ref")
            if isinstance(ref, str) and ref.startswith("state."):
                refs.append(ref)
        else:
            for v in obj.values():
                refs.extend(_collect_state_refs(v))
    elif isinstance(obj, list):
        for v in obj:
            refs.extend(_collect_state_refs(v))
    return refs


def _collect_list_state_ref_groups(obj: Any, groups: List[List[str]]) -> List[str]:
    refs: List[str] = []
    if isinstance(obj, dict):
        ref = obj.get("ref")
        if isinstance(ref, str) and ref.startswith("state."):
            return [ref]
        for value in obj.values():
            refs.extend(_collect_list_state_ref_groups(value, groups))
        return refs
    if isinstance(obj, list):
        for value in obj:
            refs.extend(_collect_list_state_ref_groups(value, groups))
        if refs:
            groups.append(refs)
        return refs
    return refs


def _validate_spec(spec: dict) -> None:
    nodes = spec.get("nodes", [])
    node_ids = {n.get("id") for n in nodes}

    entry = spec.get("entry")
    assert entry in node_ids, f"entry not found in nodes: {entry}"

    for e in spec.get("edges", []):
        assert e.get("from") in node_ids, f"edge.from missing: {e}"
        assert e.get("to") in node_ids, f"edge.to missing: {e}"

    # Validate output refs
    for ref in _collect_state_refs(spec.get("outputs", {})):
        node_id = ref.split(".", 2)[1]
        assert node_id in node_ids, f"output ref missing node: {ref}"

    # Validate node IO refs
    for node in nodes:
        io = node.get("io")
        if not io:
            continue
        for ref in _collect_state_refs(io):
            node_id = ref.split(".", 2)[1]
            assert node_id in node_ids, f"node io ref missing node: {ref}"

    # Validate output nodes are reachable from entry.
    entry = spec.get("entry")
    adj = {}
    for e in spec.get("edges", []):
        adj.setdefault(e.get("from"), []).append(e.get("to"))
    reachable = set()
    stack = [entry]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in adj.get(cur, []):
            if nxt not in reachable:
                stack.append(nxt)

    for ref in _collect_state_refs(spec.get("outputs", {})):
        node_id = ref.split(".", 2)[1]
        assert node_id in reachable, f"output ref node is unreachable from entry: {ref}"


def test_math_structure_ids_are_valid():
    workflow = get_workflow_module("math")
    spec = workflow.compile()
    for structure in workflow.enumerate_structures():
        pruned = apply_structure(spec, structure, "math")
        _validate_spec(pruned)


def test_hotpotqa_structure_ids_are_valid():
    workflow = get_workflow_module("hotpotqa")
    spec = workflow.compile()
    for structure in workflow.enumerate_structures():
        pruned = apply_structure(spec, structure, "hotpotqa")
        _validate_spec(pruned)


def test_livecodebench_structure_ids_are_valid():
    workflow = get_workflow_module("livecodebench")
    spec = workflow.compile()
    for structure in workflow.enumerate_structures():
        pruned = apply_structure(spec, structure, "livecodebench")
        _validate_spec(pruned)


def test_hotpotqa_single_generate_without_ensemble_keeps_path_to_format_answer():
    workflow = get_workflow_module("hotpotqa")
    spec = workflow.compile()
    target = None
    for structure in workflow.enumerate_structures():
        counts = structure.get("active_agent_counts") or {}
        if int(counts.get("answer_generate", 0)) == 1 and int(counts.get("sc_ensemble", 0)) == 0 and int(counts.get("format_answer", 0)) == 1:
            target = structure
            break
    assert target is not None

    pruned = apply_structure(spec, target, "hotpotqa")
    id_to_name = {node.get("id"): node.get("name") for node in pruned.get("nodes", [])}
    edge_name_pairs = {
        (id_to_name.get(edge.get("from")), id_to_name.get(edge.get("to")))
        for edge in pruned.get("edges", [])
    }
    assert ("answer_generate", "format_answer") in edge_name_pairs


def test_sc_ensemble_only_kept_for_multi_branch_inputs_across_workflows():
    for workflow_type in ("math", "hotpotqa", "livecodebench"):
        workflow = get_workflow_module(workflow_type)
        spec = workflow.compile()
        for structure in workflow.enumerate_structures():
            counts = structure.get("active_agent_counts") or {}
            if int(counts.get("sc_ensemble", 0)) <= 0:
                continue

            pruned = apply_structure(spec, structure, workflow_type)
            node_by_id = {node.get("id"): node for node in pruned.get("nodes", [])}
            sc_node = next(
                (node for node in pruned.get("nodes", []) if node.get("name") == "sc_ensemble"),
                None,
            )
            assert sc_node is not None

            groups: List[List[str]] = []
            inputs = (sc_node.get("io") or {}).get("inputs", {})
            _collect_list_state_ref_groups(inputs, groups)

            saw_agent_group = False
            for group in groups:
                agent_refs = {
                    ref.split(".", 2)[1]
                    for ref in group
                    if len(ref.split(".", 2)) >= 2
                    and (node_by_id.get(ref.split(".", 2)[1], {}).get("type") == "agent")
                }
                if not agent_refs:
                    continue
                saw_agent_group = True
                assert len(agent_refs) >= 2
            assert saw_agent_group
