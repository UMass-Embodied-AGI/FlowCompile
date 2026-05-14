from __future__ import annotations

from flowcompile.dsl.structures import _remove_nodes


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


def test_remove_nodes_drops_and_dedupes_list_refs():
    spec = {
        "entry": "a",
        "nodes": [
            {"id": "a", "name": "a", "type": "agent"},
            {"id": "b", "name": "b", "type": "agent"},
            {"id": "c", "name": "c", "type": "agent"},
            {
                "id": "chooser",
                "name": "chooser",
                "type": "agent",
                "io": {
                    "inputs": {
                        "solutions": [
                            {"ref": "state.a"},
                            {"ref": "state.b"},
                            {"ref": "state.c"},
                        ]
                    }
                },
            },
        ],
        "edges": [
            {"from": "a", "to": "b"},
            {"from": "a", "to": "c"},
            {"from": "b", "to": "chooser"},
            {"from": "c", "to": "chooser"},
        ],
        "outputs": {"final_answer": {"ref": "state.chooser"}},
    }

    pruned = _remove_nodes(spec, {"b", "c"})
    chooser = next(node for node in pruned["nodes"] if node["id"] == "chooser")
    assert chooser["io"]["inputs"]["solutions"] == [{"ref": "state.a"}]
