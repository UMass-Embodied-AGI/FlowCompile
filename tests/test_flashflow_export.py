from __future__ import annotations

from workflow_compiler.runtime.export import export_flashflow_dag


def test_export_flashflow_dag_by_budget_preset_embeds_backend_metadata():
    compiled_payload = {
        "workflow_type": "math",
        "configs": [
            {
                "config_id": "cfg_0000",
                "workflow_type": "math",
                "agents": {
                    "programmer": {"setting": "qwen35-4b_budget_2000"},
                    "refine_solver": {"setting": "gpt-5-mini_budget_unlimited"},
                    "detailed_solver": {"setting": "qwen35-4b_budget_2000"},
                    "generate_solver": {"setting": "qwen35-4b_budget_2000"},
                    "sc_ensemble": {"setting": "gpt-5-mini_budget_unlimited"},
                },
            }
        ],
        "runtime_budget_presets": {
            "low": {
                    "config_id": "cfg_0000",
                    "workflow_type": "math",
                    "agents": {
                    "programmer": {"setting": "qwen35-4b_budget_2000"},
                    "refine_solver": {"setting": "gpt-5-mini_budget_unlimited"},
                    "detailed_solver": {"setting": "qwen35-4b_budget_2000"},
                    "generate_solver": {"setting": "qwen35-4b_budget_2000"},
                    "sc_ensemble": {"setting": "gpt-5-mini_budget_unlimited"},
                },
            }
        },
    }
    model_config = {
        "models": {
            "qwen35-4b": {
                "api_type": "openai",
                "hf_model_name": "Qwen/Qwen3.5-4B",
                "enable_thinking_budget": True,
            },
            "gpt-5-mini": {
                "api_type": "azure",
                "azure_endpoint": "https://example.openai.azure.com",
                "azure_deployment": "gpt-5-mini-2024-07-18",
                "api_key": "secret",
                "api_version": "2024-10-21",
            },
        }
    }

    exported, summary = export_flashflow_dag(
        compiled_payload=compiled_payload,
        model_config=model_config,
        budget_preset="low",
    )

    assert summary["config_id"] == "cfg_0000"
    assert summary["selected_budget_preset"] == "low"
    flashflow_meta = exported["metadata"]["flashflow"]
    assert flashflow_meta["models"]["qwen35-4b"]["backend"] == "vllm"
    assert flashflow_meta["models"]["gpt-5-mini"]["backend"] == "azure"
    assert flashflow_meta["aliases"]["qwen35-4b_budget_2000"]["budget"] == 2000
    assert flashflow_meta["aliases"]["gpt-5-mini_budget_unlimited"]["budget"] == "unlimited"

    agent_nodes = {
        node["name"]: node
        for node in exported["nodes"]
        if node.get("type") == "agent"
    }
    assert agent_nodes["programmer"]["llm_ref"] == "qwen35-4b_budget_2000"
    assert agent_nodes["refine_solver"]["metadata"]["flashflow"]["backend"] == "azure"
