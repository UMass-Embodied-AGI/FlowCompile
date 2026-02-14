from __future__ import annotations

from workflow_compiler.dsl.structures import _remove_nodes


def test_remove_nodes_rewrites_output_ref_from_graph_predecessor():
    spec = {
        "entry": "a",
        "nodes": [
            {"id": "a", "name": "a", "type": "agent"},
            {"id": "b", "name": "b", "type": "agent"},
            {"id": "c", "name": "c", "type": "agent"},
            {"id": "d", "name": "d", "type": "agent"},
        ],
        "edges": [
            {"from": "a", "to": "c"},
            {"from": "c", "to": "d"},
            {"from": "b", "to": "d"},
        ],
        "outputs": {"final_answer": {"ref": "state.c"}},
    }

    # Remove c. Its real upstream input is a; b is only a sibling predecessor in node order.
    pruned = _remove_nodes(spec, {"c"})

    assert pruned["outputs"]["final_answer"]["ref"] == "state.a"
    assert {"from": "a", "to": "d"} in pruned["edges"]
