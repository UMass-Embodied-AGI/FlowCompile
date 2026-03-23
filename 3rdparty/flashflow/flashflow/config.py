"""FlashFlow exported DAG loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_workflow_dag(path: str) -> Dict[str, Any]:
    dag_path = Path(path).expanduser().resolve()
    if not dag_path.exists():
        raise FileNotFoundError(f"Workflow DAG file not found: {dag_path}")
    with open(dag_path, "r", encoding="utf-8") as f:
        if dag_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(f) or {}
        else:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("FlashFlow workflow DAG must be a mapping.")
    return payload


def get_flashflow_metadata(dag: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dag.get("metadata") or {}
    flashflow = metadata.get("flashflow") or {}
    if not isinstance(flashflow, dict) or not flashflow:
        raise ValueError("Workflow DAG is missing metadata.flashflow export metadata.")
    aliases = flashflow.get("aliases") or {}
    models = flashflow.get("models") or {}
    if not aliases or not models:
        raise ValueError("Workflow DAG flashflow metadata must include aliases and models.")
    return flashflow
