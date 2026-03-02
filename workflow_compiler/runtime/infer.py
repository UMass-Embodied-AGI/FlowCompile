"""Unified runtime infer API."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from workflow_compiler.runtime.engine import run_batch_sync
from workflow_compiler.runtime.selector import select_config


def _normalize_query(
    query: Any,
    workflow_type: str,
    query_id: Optional[str] = None,
) -> Dict[str, Any]:
    if isinstance(query, dict):
        payload = dict(query)
    else:
        key = "question" if workflow_type in {"hotpotqa", "livecodebench"} else "problem"
        payload = {key: str(query)}
    if query_id is not None and not any(k in payload for k in ("id", "_id", "question_id", "unique_id")):
        payload["id"] = str(query_id)
    return payload


def select_runtime_config(
    configs: List[Dict[str, Any]],
    strategy: str = "preference",
    budget: float = 0.5,
    min_accuracy: Optional[float] = None,
    max_latency: Optional[float] = None,
) -> Dict[str, Any]:
    selected = select_config(
        configs,
        strategy=strategy,
        budget=budget,
        min_accuracy=min_accuracy,
        max_latency=max_latency,
    )
    if not selected:
        raise SystemExit("No runtime config matched the selection criteria.")
    return selected


def _format_runtime_record_single(
    query: Dict[str, Any],
    selected_config: Dict[str, Any],
    execution: Dict[str, Any],
    routing_metadata: Optional[Dict[str, Any]] = None,
    routing_runtime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    record = {
        "query": query,
        "selected_config": selected_config,
        "answer": execution.get("output"),
        "workflow_output": execution.get("output"),
        "routing_runtime_seconds": routing_runtime_seconds,
        "actual_runtime_seconds": execution.get("actual_runtime_seconds"),
        "query_id": execution.get("query_id"),
        "config_id": execution.get("config_id"),
        "structure_id": execution.get("structure_id"),
        "output_dir": execution.get("output_dir"),
    }
    if routing_metadata is not None:
        record["routing_metadata"] = routing_metadata
    return record


def _format_runtime_record_batch(
    query: Dict[str, Any],
    selected_config: Dict[str, Any],
    execution: Dict[str, Any],
    routing_metadata: Optional[Dict[str, Any]] = None,
    routing_runtime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    record = {
        "query": query,
        "selected_config": selected_config,
        "answer": execution.get("output"),
        "routing_runtime_seconds": routing_runtime_seconds,
        "query_id": execution.get("query_id"),
        "config_id": execution.get("config_id"),
        "structure_id": execution.get("structure_id"),
        "output_dir": execution.get("output_dir"),
    }
    if routing_metadata is not None:
        record["routing_metadata"] = routing_metadata
    return record


def _select_knn_router_config(
    query_payload: Dict[str, Any],
    workflow_type: str,
    router: Any,
    budget: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    if router is None:
        raise SystemExit("strategy knn-router requires a fitted router.")
    start_time = time.perf_counter()
    configs, routing_metadata = router.build_runtime_candidates(query_payload, workflow_type)
    selected = select_runtime_config(
        configs,
        strategy="preference",
        budget=budget,
    )
    routing_runtime_seconds = time.perf_counter() - start_time
    return selected, routing_metadata, routing_runtime_seconds


def infer_runtime(
    query: Any,
    configs: List[Dict[str, Any]],
    workflow_type: str,
    output_dir: Path,
    strategy: str = "preference",
    budget: float = 0.5,
    min_accuracy: Optional[float] = None,
    max_latency: Optional[float] = None,
    query_id: Optional[str] = None,
    router: Any = None,
) -> Dict[str, Any]:
    query_payload = _normalize_query(query, workflow_type, query_id=query_id)
    routing_metadata = None
    routing_runtime_seconds = None
    if strategy == "knn-router":
        selected, routing_metadata, routing_runtime_seconds = _select_knn_router_config(
            query_payload=query_payload,
            workflow_type=workflow_type,
            router=router,
            budget=budget,
        )
    else:
        selected = select_runtime_config(
            configs,
            strategy=strategy,
            budget=budget,
            min_accuracy=min_accuracy,
            max_latency=max_latency,
        )
    execution = run_batch_sync([(query_payload, selected)], workflow_type, output_dir)[0]
    return _format_runtime_record_single(
        query_payload,
        selected,
        execution,
        routing_metadata=routing_metadata,
        routing_runtime_seconds=routing_runtime_seconds,
    )


def infer_runtime_batch(
    queries: List[Dict[str, Any]],
    configs: List[Dict[str, Any]],
    workflow_type: str,
    output_dir: Path,
    strategy: str = "preference",
    budget: float = 0.5,
    min_accuracy: Optional[float] = None,
    max_latency: Optional[float] = None,
    router: Any = None,
) -> List[Dict[str, Any]]:
    pairs = []
    routing_metadata_by_query: List[Optional[Dict[str, Any]]] = []
    routing_runtime_by_query: List[Optional[float]] = []
    for raw_query in queries:
        query_payload = _normalize_query(raw_query, workflow_type)
        if strategy == "knn-router":
            selected, routing_metadata, routing_runtime_seconds = _select_knn_router_config(
                query_payload=query_payload,
                workflow_type=workflow_type,
                router=router,
                budget=budget,
            )
        else:
            selected = select_runtime_config(
                configs,
                strategy=strategy,
                budget=budget,
                min_accuracy=min_accuracy,
                max_latency=max_latency,
            )
            routing_metadata = None
            routing_runtime_seconds = None
        pairs.append((query_payload, selected))
        routing_metadata_by_query.append(routing_metadata)
        routing_runtime_by_query.append(routing_runtime_seconds)
    executions = run_batch_sync(pairs, workflow_type, output_dir)
    return [
        _format_runtime_record_batch(
            query,
            selected,
            execution,
            routing_metadata=routing_metadata,
            routing_runtime_seconds=routing_runtime_seconds,
        )
        for ((query, selected), execution, routing_metadata, routing_runtime_seconds) in zip(
            pairs,
            executions,
            routing_metadata_by_query,
            routing_runtime_by_query,
        )
    ]
