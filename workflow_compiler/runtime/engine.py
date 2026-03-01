"""Runtime execution engine for FlowCompile."""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import asyncio
import time

from workflow_compiler.core.llm.config import build_setting
from workflow_compiler.dsl.runtime import run_dsl_query


def _agent_setting(agent_info: Dict[str, Any]) -> Optional[str]:
    setting = agent_info.get("setting")
    if setting:
        return setting
    return build_setting(agent_info.get("model"), agent_info.get("budget"))


def _build_llm_configs(config: Dict[str, Any], workflow_type: str) -> Dict[str, Any]:
    agents = config.get("agents") or {}

    if workflow_type in ("math", "gsm8k"):
        sc_ensemble = _agent_setting(agents.get("sc_ensemble", {}))
        llm_configs = {
            "meta": sc_ensemble,
            "programmer": _agent_setting(agents.get("programmer", {})),
            "sc_ensemble": sc_ensemble,
            "refine_solver": _agent_setting(agents.get("refine_solver", {})),
            "detailed_solver": _agent_setting(agents.get("detailed_solver", {})),
            "generate_solver": _agent_setting(agents.get("generate_solver", {})),
        }
        return llm_configs

    if workflow_type == "hotpotqa":
        sc_ensemble = _agent_setting(agents.get("sc_ensemble", {}))
        return {
            "meta": sc_ensemble,
            "answer_generate": _agent_setting(agents.get("answer_generate", {})),
            "sc_ensemble": sc_ensemble,
            "format_answer": _agent_setting(agents.get("format_answer", {})),
        }

    if workflow_type == "livecodebench":
        sc_ensemble = _agent_setting(agents.get("sc_ensemble", {}))
        reflection = _agent_setting(agents.get("reflection_test", {}))
        code_gen = _agent_setting(agents.get("code_generate", {}))
        test_setting = _agent_setting(agents.get("test", {})) or reflection or code_gen or sc_ensemble
        return {
            "meta": sc_ensemble,
            "code_generate": code_gen,
            "sc_ensemble": sc_ensemble,
            "test": test_setting,
            "reflection_test": reflection,
        }

    raise ValueError(f"Unsupported workflow_type: {workflow_type}")


async def run_query(
    query: Dict[str, Any],
    config: Dict[str, Any],
    workflow_type: str,
    output_dir: Path,
) -> Dict[str, Any]:
    if not config:
        raise ValueError("No configuration provided for runtime execution")
    if not isinstance(query, dict):
        query = {"problem": str(query)}
    # Derive query id and content
    query_id = query.get("id") or query.get("_id") or query.get("question_id") or query.get("unique_id")
    if query_id is None:
        query_id = f"query_{abs(hash(str(query))) % 100000}"

    run_dir = output_dir / str(query_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    structure_id = config.get("structure_id")

    # DSL runtime path for supported workflows
    if workflow_type in ("math", "gsm8k", "hotpotqa", "livecodebench"):
        start_time = time.perf_counter()
        output = await run_dsl_query(query, config, workflow_type, run_dir)
        elapsed_seconds = time.perf_counter() - start_time
        return {
            "query_id": str(query_id),
            "output": output,
            "structure_id": structure_id,
            "config_id": config.get("config_id"),
            "output_dir": str(run_dir),
            "actual_runtime_seconds": elapsed_seconds,
        }

    raise ValueError(f"Unsupported workflow_type: {workflow_type}")


async def run_batch(
    query_config_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    workflow_type: str,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    results = []
    for query, config in query_config_pairs:
        result = await run_query(query, config, workflow_type, output_dir)
        results.append(result)
    return results


def run_batch_sync(
    query_config_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    workflow_type: str,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    return asyncio.run(run_batch(query_config_pairs, workflow_type, output_dir))
