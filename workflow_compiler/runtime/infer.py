"""Unified runtime infer API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
    alpha: float = 0.5,
    min_accuracy: Optional[float] = None,
    max_latency: Optional[float] = None,
) -> Dict[str, Any]:
    selected = select_config(
        configs,
        strategy=strategy,
        alpha=alpha,
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
) -> Dict[str, Any]:
    return {
        "query": query,
        "selected_config": selected_config,
        "answer": execution.get("output"),
        "workflow_output": execution.get("output"),
        "actual_runtime_seconds": execution.get("actual_runtime_seconds"),
        "query_id": execution.get("query_id"),
        "config_id": execution.get("config_id"),
        "structure_id": execution.get("structure_id"),
        "output_dir": execution.get("output_dir"),
    }


def _format_runtime_record_batch(
    query: Dict[str, Any],
    selected_config: Dict[str, Any],
    execution: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "query": query,
        "selected_config": selected_config,
        "answer": execution.get("output"),
        "query_id": execution.get("query_id"),
        "config_id": execution.get("config_id"),
        "structure_id": execution.get("structure_id"),
        "output_dir": execution.get("output_dir"),
    }


def infer_runtime(
    query: Any,
    configs: List[Dict[str, Any]],
    workflow_type: str,
    output_dir: Path,
    strategy: str = "preference",
    alpha: float = 0.5,
    min_accuracy: Optional[float] = None,
    max_latency: Optional[float] = None,
    query_id: Optional[str] = None,
) -> Dict[str, Any]:
    selected = select_runtime_config(
        configs,
        strategy=strategy,
        alpha=alpha,
        min_accuracy=min_accuracy,
        max_latency=max_latency,
    )
    query_payload = _normalize_query(query, workflow_type, query_id=query_id)
    execution = run_batch_sync([(query_payload, selected)], workflow_type, output_dir)[0]
    return _format_runtime_record_single(query_payload, selected, execution)


def infer_runtime_batch(
    queries: List[Dict[str, Any]],
    configs: List[Dict[str, Any]],
    workflow_type: str,
    output_dir: Path,
    strategy: str = "preference",
    alpha: float = 0.5,
    min_accuracy: Optional[float] = None,
    max_latency: Optional[float] = None,
) -> List[Dict[str, Any]]:
    selected = select_runtime_config(
        configs,
        strategy=strategy,
        alpha=alpha,
        min_accuracy=min_accuracy,
        max_latency=max_latency,
    )
    pairs = [(_normalize_query(q, workflow_type), selected) for q in queries]
    executions = run_batch_sync(pairs, workflow_type, output_dir)
    return [
        _format_runtime_record_batch(query, selected, execution)
        for (query, _), execution in zip(pairs, executions)
    ]
