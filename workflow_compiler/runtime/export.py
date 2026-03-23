"""Export FlowCompile DAGs for FlashFlow serving."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from workflow_compiler.core.llm.config import load_model_config_payload, parse_config
from workflow_compiler.dsl.structures import apply_structure
from workflow_compiler.runtime.selector import (
    RUNTIME_PREFERENCE_BUDGET_PRESETS,
    select_config,
)
from workflow_compiler.workflows.dsl_registry import get_workflow_module


FLASHFLOW_EXPORT_SCHEMA_VERSION = "flashflow.export.v1"


def _resolve_compiled_config(
    compiled_payload: Dict[str, Any],
    *,
    config_id: Optional[str] = None,
    budget_preset: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    configs = list(compiled_payload.get("configs") or [])
    all_configs = list(compiled_payload.get("all_configs") or [])
    runtime_budget_presets = compiled_payload.get("runtime_budget_presets") or {}

    if config_id:
        for candidate in configs + all_configs:
            if str(candidate.get("config_id")) == str(config_id):
                return candidate, None
        raise ValueError(f"Unknown config_id '{config_id}'.")

    if not budget_preset:
        raise ValueError("Either config_id or budget_preset is required.")

    preset_name = str(budget_preset).strip().lower()
    if preset_name not in RUNTIME_PREFERENCE_BUDGET_PRESETS:
        raise ValueError(
            f"Unknown budget preset '{budget_preset}'. "
            f"Expected one of {sorted(RUNTIME_PREFERENCE_BUDGET_PRESETS.keys())}."
        )

    preset_config = runtime_budget_presets.get(preset_name)
    if isinstance(preset_config, dict) and preset_config:
        return preset_config, preset_name

    selected = select_config(
        configs,
        strategy="preference",
        budget=float(RUNTIME_PREFERENCE_BUDGET_PRESETS[preset_name]),
    )
    if not selected:
        raise ValueError(f"No config available for budget preset '{preset_name}'.")
    return selected, preset_name


def _derive_flashflow_backend(model_cfg: Dict[str, Any]) -> str:
    api_type = str(model_cfg.get("api_type") or "openai").lower()
    if api_type == "azure":
        return "azure"
    return "vllm"


def _derive_flashflow_strategy(model_cfg: Dict[str, Any], backend: str) -> str:
    strategy = (
        model_cfg.get("flashflow_thinking_strategy")
        or model_cfg.get("thinking_budget_strategy")
        or model_cfg.get("thinking_budget_impl")
    )
    if strategy:
        return str(strategy)
    if backend == "azure":
        return "unlimited_only"
    if bool(model_cfg.get("enable_thinking_budget", False)):
        return "vllm_plugin"
    return "unlimited_only"


def _build_model_metadata(model_name: str, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    backend = _derive_flashflow_backend(model_cfg)
    metadata: Dict[str, Any] = {
        "name": model_name,
        "backend": backend,
        "api_type": str(model_cfg.get("api_type") or "openai").lower(),
        "default_thinking_strategy": _derive_flashflow_strategy(model_cfg, backend),
        "enable_thinking_budget": bool(model_cfg.get("enable_thinking_budget", False)),
        "raw": copy.deepcopy(model_cfg),
    }
    if backend == "azure":
        metadata.update(
            {
                "azure_endpoint": model_cfg.get("azure_endpoint") or model_cfg.get("endpoint"),
                "azure_deployment": (
                    model_cfg.get("azure_deployment")
                    or model_cfg.get("deployment_name")
                    or model_cfg.get("deployment")
                    or model_cfg.get("model")
                    or model_name
                ),
                "api_version": model_cfg.get("api_version"),
                "api_key": model_cfg.get("api_key") or model_cfg.get("key"),
            }
        )
    else:
        metadata.update(
            {
                "artifact_id": model_cfg.get("hf_model_name") or model_cfg.get("model") or model_name,
                "tokenizer": (
                    model_cfg.get("tokenizer")
                    or model_cfg.get("hf_model_name")
                    or model_cfg.get("model")
                    or model_name
                ),
                "thinking_budget_reasoning_parser": model_cfg.get("thinking_budget_reasoning_parser"),
            }
        )
    return metadata


def export_flashflow_dag(
    *,
    compiled_payload: Dict[str, Any],
    model_config: Optional[Dict[str, Any]] = None,
    workflow_type: Optional[str] = None,
    config_id: Optional[str] = None,
    budget_preset: Optional[str] = None,
    openclaw_lobster_workflow_file: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    selected_config, selected_preset = _resolve_compiled_config(
        compiled_payload,
        config_id=config_id,
        budget_preset=budget_preset,
    )
    workflow_type = (
        workflow_type
        or selected_config.get("workflow_type")
        or compiled_payload.get("workflow_type")
    )
    if not workflow_type:
        raise ValueError("workflow_type is required to export FlashFlow DAG.")

    payload = load_model_config_payload(model_config)
    models_payload = payload.get("models") or {}
    metadata = compiled_payload.get("metadata") or {}
    openclaw_workflow_file = (
        openclaw_lobster_workflow_file
        or metadata.get("openclaw_lobster_workflow_file")
    )
    workflow_module = get_workflow_module(
        workflow_type,
        openclaw_lobster_workflow_file=openclaw_workflow_file,
    )
    spec = workflow_module.compile()

    structure_id = selected_config.get("structure_id")
    if structure_id:
        structure = workflow_module.get_structure(structure_id)
        spec = apply_structure(spec, structure, workflow_type)

    agents_cfg = selected_config.get("agents") or {}
    alias_map: Dict[str, Dict[str, Any]] = {}
    models_meta: Dict[str, Dict[str, Any]] = {}
    node_models: Dict[str, Dict[str, Any]] = {}

    exported = copy.deepcopy(spec)
    exported_metadata = dict(exported.get("metadata") or {})

    for node in exported.get("nodes") or []:
        if node.get("type") != "agent":
            continue
        agent_name = str(node.get("name") or "")
        agent_info = agents_cfg.get(agent_name) or {}
        setting = agent_info.get("setting")
        if not setting:
            model = agent_info.get("model")
            budget = agent_info.get("budget")
            if model:
                setting = str(model) if budget is None else f"{model}_budget_{budget}"
        if not setting:
            raise ValueError(
                f"Selected config is missing a concrete setting for agent '{agent_name}'."
            )
        model_name, budget = parse_config(str(setting))
        if not model_name:
            raise ValueError(f"Could not parse model alias '{setting}' for agent '{agent_name}'.")
        model_cfg = models_payload.get(model_name)
        if not isinstance(model_cfg, dict):
            raise ValueError(
                f"Model '{model_name}' used by agent '{agent_name}' is missing from model_config."
            )
        backend = _derive_flashflow_backend(model_cfg)
        if model_name not in models_meta:
            models_meta[model_name] = _build_model_metadata(model_name, model_cfg)
        alias_map[str(setting)] = {
            "model_alias": str(setting),
            "base_model": model_name,
            "budget": budget,
            "backend": backend,
            "default_thinking_strategy": models_meta[model_name]["default_thinking_strategy"],
        }
        node["llm_ref"] = str(setting)
        node_meta = dict(node.get("metadata") or {})
        node_meta["flashflow"] = {
            "model_alias": str(setting),
            "base_model": model_name,
            "backend": backend,
        }
        node["metadata"] = node_meta
        node_models[str(node.get("id") or agent_name)] = node_meta["flashflow"]

    exported_metadata["flashflow"] = {
        "schema_version": FLASHFLOW_EXPORT_SCHEMA_VERSION,
        "workflow_type": workflow_type,
        "config_id": selected_config.get("config_id"),
        "selected_budget_preset": selected_preset,
        "structure_id": structure_id,
        "models": models_meta,
        "aliases": alias_map,
        "nodes": node_models,
    }
    exported["metadata"] = exported_metadata
    exported["flashflow_schema_version"] = FLASHFLOW_EXPORT_SCHEMA_VERSION
    return exported, {
        "config_id": selected_config.get("config_id"),
        "selected_budget_preset": selected_preset,
        "structure_id": structure_id,
        "workflow_type": workflow_type,
    }


def write_flashflow_dag(
    output_path: Path,
    exported_dag: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(exported_dag, f, indent=2)
