"""Runtime router utilities (KNN routing)."""
from __future__ import annotations

from typing import Dict, Any, List, Optional

from workflow_compiler.routers import get_router
from workflow_compiler.routers.utils import load_consolidated_data
from workflow_compiler.core.llm.config import parse_config


def _pareto_config_to_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    # Convert ParetoConfiguration dict to runtime config schema
    agents: Dict[str, Any] = {}
    for agent_key, setting in (config.get("subagent_settings") or {}).items():
        model, budget = parse_config(setting)
        agents[agent_key] = {
            "setting": setting,
            "model": model,
            "budget": budget,
        }

    return {
        "config_id": config.get("workflow_id"),
        "structure_id": config.get("structure_id"),
        "agents": agents,
        "metrics": {
            "expected_accuracy": config.get("expected_accuracy"),
            "expected_latency": config.get("expected_latency"),
        },
        "pareto": {
            "is_pareto": True,
            "rank": 0,
        },
    }


def build_knn_router(
    query_data_file: str,
    k: int = 10,
    embedding_model: str = "allenai/longformer-base-4096",
    max_length: int = 4096,
    embedding_batch_size: int = 8,
    embedding_cache_file: Optional[str] = None,
    accuracy_thresholds: Optional[List[float]] = None,
    search_space: Optional[Dict[str, Any]] = None,
):
    router = get_router(
        "knn",
        k=k,
        embedding_model=embedding_model,
        max_length=max_length,
        embedding_batch_size=embedding_batch_size,
        embedding_cache_file=embedding_cache_file,
        accuracy_thresholds=accuracy_thresholds,
        search_space=search_space,
    )
    query_data_table = load_consolidated_data(query_data_file)
    router.fit_from_query_table(query_data_table)
    return router


def route_query(router, query: Dict[str, Any], workflow_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
    result = router.route(query=query, workflow_type=workflow_type, top_k=top_k)
    pareto_configs = result.metadata.get("pareto_configs", []) if result.metadata else []
    return [_pareto_config_to_runtime(cfg) for cfg in pareto_configs]
