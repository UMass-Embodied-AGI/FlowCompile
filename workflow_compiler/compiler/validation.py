#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run evaluations for workflow configurations from JSON file.
Evaluates each configuration and saves actual accuracy results.
"""

import json
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm.asyncio import tqdm

# Import the evaluation components
from workflow_compiler.dsl.runtime import DslWorkflowRunner
from workflow_compiler.dsl.structures import apply_structure
from workflow_compiler.workflows.dsl_registry import get_workflow_module
from workflow_compiler.benchmarks import get_benchmark, get_benchmark_info
from workflow_compiler.core.logs import logger
from workflow_compiler.core.data_paths import resolve_existing_path

_LEGACY_TRACE_FIELDS = {
    "is_correct",
    "f1_score",
    "pass_at_1",
    "private_test_passed",
    "extracted_answer",
    "test_passed",
    "correct",
}

_ACTIVE_LLM_REFS_CACHE: Dict[Tuple[str, Optional[str]], List[str]] = {}


def _resolve_benchmark(dataset: str) -> Dict[str, Any]:
    info = get_benchmark_info(dataset)
    benchmark_class = info["class"]
    info["canonical_dataset_name"] = info["name"]
    info["workflow_type"] = info.get("workflow_type") or getattr(benchmark_class, "WORKFLOW_TYPE", "math")
    info["metric_name"] = info.get("metric_name") or getattr(benchmark_class, "METRIC_NAME", "accuracy")
    return info


def _default_data_path(dataset: str, split: str) -> Optional[str]:
    info = _resolve_benchmark(dataset)
    defaults = info.get("default_split_paths") or {}
    path = defaults.get(split)
    return resolve_existing_path(path) or path


def _dataset_to_workflow_type(dataset: str) -> str:
    return _resolve_benchmark(dataset)["workflow_type"]


def _metric_for_dataset(dataset: str) -> str:
    return _resolve_benchmark(dataset)["metric_name"]


def _score_is_success(metric: str, score: float) -> bool:
    if metric == "f1":
        return score > 0.5
    return score >= 0.5


def _result_score(dataset: str, result: Any) -> float:
    benchmark_class = _resolve_benchmark(dataset)["class"]
    if hasattr(benchmark_class, "score_from_result"):
        return float(benchmark_class.score_from_result(result))
    return float(result[-1])


def _result_key(dataset: str, result: Any) -> Optional[str]:
    benchmark_class = _resolve_benchmark(dataset)["class"]
    if hasattr(benchmark_class, "result_key"):
        value = benchmark_class.result_key(result)
        if value is None:
            return None
        return str(value)
    if result:
        return str(result[0])
    return None


def _trace_key(dataset: str, trace: Dict[str, Any]) -> Optional[str]:
    benchmark_class = _resolve_benchmark(dataset)["class"]
    if hasattr(benchmark_class, "trace_key"):
        value = benchmark_class.trace_key(trace)
        if value is None:
            return None
        return str(value)
    orig = trace.get("metadata", {}).get("original_sample", {}) or {}
    problem = orig.get("problem") or orig.get("question") or trace.get("problem")
    return str(problem) if problem is not None else None


def annotate_trace_file_with_scores(
    trace_file: Path,
    dataset: str,
    results: List[Any],
) -> None:
    if not trace_file.exists():
        logger.warning(f"Trace file not found for annotation: {trace_file}")
        return

    metric = _metric_for_dataset(dataset)

    # Build key -> scores list mapping (handles duplicates)
    score_map: Dict[str, List[float]] = {}
    fallback_scores: List[float] = []
    for result in results:
        score = _result_score(dataset, result)
        fallback_scores.append(score)
        key = _result_key(dataset, result)
        if key is None:
            continue
        score_map.setdefault(key, []).append(score)

    # Read traces
    traces: List[Dict[str, Any]] = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    # Annotate
    for trace in traces:
        key = _trace_key(dataset, trace)
        score: Optional[float] = None
        if key is not None and key in score_map and score_map[key]:
            score = score_map[key].pop(0)
        elif fallback_scores:
            score = fallback_scores.pop(0)

        # Remove legacy evaluation fields
        for legacy_key in _LEGACY_TRACE_FIELDS:
            trace.pop(legacy_key, None)

        if score is not None:
            trace["score"] = float(score)
            trace["metric"] = metric
        else:
            logger.warning(f"Unable to map score for trace entry (dataset={dataset}).")

    # Write back (atomic)
    tmp_file = trace_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    tmp_file.replace(trace_file)

def _load_existing_results(
    config_idx: int,
    output_base_dir: Path,
    expected_config_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check if configuration has already been evaluated and load results."""
    config_dir = output_base_dir / f"config_{config_idx:04d}"
    config_results_file = config_dir / "config_results.json"
    
    if config_results_file.exists():
        try:
            with open(config_results_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if expected_config_id:
                cached_config_id = payload.get("_compiled_config_id") or payload.get("config_id")
                if cached_config_id and str(cached_config_id) != str(expected_config_id):
                    logger.warning(
                        f"Cached result mismatch for config_{config_idx:04d}: "
                        f"expected config_id={expected_config_id}, found={cached_config_id}. Re-evaluating."
                    )
                    return None
            return payload
        except Exception as e:
            logger.warning(f"Failed to load existing results for config {config_idx}: {e}")
    
    return None


def _extract_predicted_latency(config: Dict[str, Any]) -> float:
    metrics = config.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("expected_latency")
        if value is not None:
            try:
                return float(value)
            except Exception as exc:
                raise ValueError(f"Invalid latency value for 'metrics.expected_latency': {value}") from exc

    for key in ("workflow_latency", "expected_latency"):
        value = config.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception as exc:
            raise ValueError(f"Invalid latency value for '{key}': {value}") from exc
    raise ValueError(
        "Missing predicted latency. Expected one of: metrics.expected_latency, workflow_latency, expected_latency"
    )


def _extract_pareto_rank(config: Dict[str, Any]) -> int:
    pareto = config.get("pareto")
    if isinstance(pareto, dict):
        value = pareto.get("rank")
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

    value = config.get("pareto_rank", 10**9)
    try:
        return int(value)
    except Exception:
        return 10**9


def _is_pareto_config(config: Dict[str, Any]) -> bool:
    pareto = config.get("pareto")
    if isinstance(pareto, dict):
        return bool(pareto.get("is_pareto", False))
    return bool(config.get("is_pareto", False))


def _sample_pareto_even_by_latency(
    configs: List[Dict[str, Any]],
    n: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if n <= 0:
        raise ValueError("pareto_sample_n must be >= 1")

    records: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for idx, config in enumerate(configs):
        latency = _extract_predicted_latency(config)
        records.append((latency, _extract_pareto_rank(config), idx, config))

    records.sort(key=lambda x: (x[0], x[1], x[2]))
    candidate_count = len(records)
    if candidate_count == 0:
        return [], {
            "method": "pareto_latency_even",
            "requested_n": int(n),
            "candidate_count": 0,
            "selected_count": 0,
            "latency_min": None,
            "latency_max": None,
        }

    if n >= candidate_count:
        selected = [entry[3] for entry in records]
    elif n == 1:
        selected = [records[0][3]]
    else:
        selected_indexes = {0, candidate_count - 1}
        min_latency = records[0][0]
        max_latency = records[-1][0]
        for i in range(1, n - 1):
            target = min_latency + (max_latency - min_latency) * (i / (n - 1))
            candidate_positions = [p for p in range(candidate_count) if p not in selected_indexes]
            if not candidate_positions:
                break
            best_pos = min(
                candidate_positions,
                key=lambda p: (
                    abs(records[p][0] - target),
                    records[p][0],
                    records[p][1],
                    records[p][2],
                ),
            )
            selected_indexes.add(best_pos)
        selected = [records[p][3] for p in sorted(selected_indexes)]

    latencies = [entry[0] for entry in records]
    metadata = {
        "method": "pareto_latency_even",
        "requested_n": int(n),
        "candidate_count": int(candidate_count),
        "selected_count": int(len(selected)),
        "latency_min": float(min(latencies)) if latencies else None,
        "latency_max": float(max(latencies)) if latencies else None,
    }
    return selected, metadata


def _pareto_rank_config_index(config: Dict[str, Any]) -> Optional[int]:
    pareto = config.get("pareto")
    if isinstance(pareto, dict) and pareto.get("rank") is not None:
        value = pareto.get("rank")
    else:
        value = config.get("pareto_rank")
    if value is None:
        return None
    try:
        idx = int(value)
    except Exception:
        return None
    if idx < 0:
        return None
    return idx


def _build_evaluation_items(configs: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """Build stable (config_index, config) pairs.

    config_index is derived from pareto_rank when possible so output folder names
    are stable and rank-aligned (e.g., rank 0 -> config_0000), regardless of
    shuffled evaluation order.
    """
    items: List[Tuple[int, Dict[str, Any]]] = []
    used_indexes = set()
    fallback_idx = 0

    for cfg in configs:
        cfg_idx = _pareto_rank_config_index(cfg)
        if cfg_idx is None or cfg_idx in used_indexes:
            while fallback_idx in used_indexes:
                fallback_idx += 1
            cfg_idx = fallback_idx
            fallback_idx += 1
        used_indexes.add(cfg_idx)
        items.append((cfg_idx, cfg))
    return items


def _save_interim_results(
    updated_configs: List[Dict[str, Any]],
    workflow_data: Dict[str, Any],
    output_base_dir: Path,
    workflow_type: str,
    sampling_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Helper function to save interim results."""
    interim_results = {
        'pareto_optimal': updated_configs,
        'non_pareto_sample': [],
        'metadata': workflow_data.get('metadata', {}),
        'evaluation_metadata': {
            'evaluation_timestamp': datetime.now().isoformat(),
            'workflow_type': workflow_type,
            'total_evaluated': len(updated_configs),
            'output_directory': str(output_base_dir),
        }
    }
    if sampling_metadata is not None:
        interim_results["evaluation_metadata"]["sampling"] = sampling_metadata
    
    interim_file = output_base_dir / "workflow_results_interim.json"
    with open(interim_file, 'w', encoding='utf-8') as f:
        json.dump(interim_results, f, indent=2)


def _build_llm_configs_for_workflow(workflow_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    del workflow_type
    llm_configs: Dict[str, Any] = {}
    agents = config.get("agents")
    if not isinstance(agents, dict):
        return llm_configs

    for agent_name, agent_info in agents.items():
        if not isinstance(agent_info, dict):
            continue
        setting = agent_info.get("setting")
        if setting:
            llm_configs[str(agent_name)] = setting
            continue
        model = agent_info.get("model")
        budget = agent_info.get("budget")
        if model:
            if budget is None:
                llm_configs[str(agent_name)] = str(model)
            else:
                llm_configs[str(agent_name)] = f"{model}_budget_{budget}"
    return llm_configs


def _active_llm_refs_for_structure(workflow_type: str, structure_id: Optional[str]) -> List[str]:
    cache_key = (workflow_type, structure_id)
    cached = _ACTIVE_LLM_REFS_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    spec = get_workflow_module(workflow_type).compile()
    structured_spec = apply_structure(spec, structure_id, workflow_type)
    refs: List[str] = []
    seen = set()
    for node in structured_spec.get("nodes", []):
        if node.get("type") != "agent":
            continue
        name = str(node.get("name") or "")
        llm_ref = node.get("llm_ref")
        if llm_ref is not None and str(llm_ref) != name:
            raise ValueError(
                "Non-canonical llm_ref override is not supported in validation: "
                f"{name}->{llm_ref}"
            )
        if name and name not in seen:
            seen.add(name)
            refs.append(name)
    refs = sorted(refs)
    _ACTIVE_LLM_REFS_CACHE[cache_key] = refs
    return list(refs)


def _validate_active_llm_refs(
    workflow_type: str,
    structure_id: Optional[str],
    llm_configs: Dict[str, Any],
    config: Dict[str, Any],
    config_idx: int,
    active_llm_refs: List[str],
) -> None:
    missing_refs = sorted(ref for ref in active_llm_refs if not llm_configs.get(ref))
    if not missing_refs:
        return

    config_id = config.get("_compiled_config_id") or config.get("config_id")
    msg = (
        "Missing LLM setting(s) for active operator(s): "
        f"{', '.join(missing_refs)} (workflow_type={workflow_type}, "
        f"structure_id={structure_id or '<default>'}, config_index={config_idx}"
    )
    if config_id:
        msg += f", config_id={config_id}"
    msg += ")."
    raise ValueError(msg)


def _attach_score_fields(target: Dict[str, Any], workflow_type: str, average_score: float) -> None:
    # Keep backward compatibility fields expected by downstream code.
    target["actual_accuracy"] = average_score
    if workflow_type == "hotpotqa":
        target["actual_f1"] = average_score
    elif workflow_type == "livecodebench":
        target["actual_pass_rate"] = average_score


def _log_workflow_llm_configs(
    workflow_type: str,
    llm_configs: Dict[str, Any],
    active_llm_refs: Optional[List[str]] = None,
) -> None:
    del workflow_type
    if active_llm_refs is None:
        keys = sorted(llm_configs.keys())
    else:
        keys = list(active_llm_refs)
    for key in keys:
        logger.info(f"{key}: {llm_configs.get(key)}")


async def evaluate_configuration(
    config: Dict[str, Any],
    config_idx: int,
    output_base_dir: Path,
    workflow_type: str = "dsl",
    split: str = "validate",
    dataset: str = "MATH",
    data_path: Optional[str] = None,
    entry_point_file: Optional[str] = None,
    max_tasks: int = 16,  # Maximum concurrent tasks for evaluation
) -> Dict[str, Any]:
    """
    Evaluate a single workflow configuration.
    
    Args:
        config: Configuration dictionary with agent settings
        config_idx: Index of configuration in the list
        output_base_dir: Base directory for all outputs
        workflow_type: Workflow engine type (DSL-only for supported datasets)
        split: Which dataset split to use ('validate' or 'test')
        dataset: Dataset name ('MATH' or 'HotpotQA')
        max_tasks: Maximum number of concurrent tasks for evaluation
    Returns:
        Updated configuration with actual accuracy
    """
    # Create config-specific output directory
    config_dir = output_base_dir / f"config_{config_idx:04d}"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract configurations - support both _cfg and _setting field names
    # _cfg is used in older configs, _setting is used in workflow_by_level configs
    # Also extract structure_id for dynamic workflow execution
    structure_id = config.get('structure_id', None)
    
    benchmark_info = _resolve_benchmark(dataset)
    workflow_type_name = benchmark_info["workflow_type"]
    llm_configs = _build_llm_configs_for_workflow(workflow_type_name, config)
    active_llm_refs = _active_llm_refs_for_structure(workflow_type_name, structure_id)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating Configuration {config_idx + 1}")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset: {dataset}")
    if structure_id:
        logger.info(f"Structure: {structure_id}")
    _log_workflow_llm_configs(workflow_type_name, llm_configs, active_llm_refs=active_llm_refs)
    logger.info(f"Output directory: {config_dir}")
    
    # Save configuration info to the config sub-folder
    config_info_file = config_dir / "config_info.json"
    config_info = {
        'config_index': config_idx,
        'original_config': config,
        'llm_configs': llm_configs,
        'active_llm_refs': active_llm_refs,
        'workflow_type': "dsl",
        'split': split,
        'dataset': dataset,
        'evaluation_start_time': datetime.now().isoformat(),
    }
    with open(config_info_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2)
    logger.info(f"Saved configuration info to: {config_info_file}")

    workflow = None
    try:
        _validate_active_llm_refs(
            workflow_type=workflow_type_name,
            structure_id=structure_id,
            llm_configs=llm_configs,
            config=config,
            config_idx=config_idx,
            active_llm_refs=active_llm_refs,
        )

        # Initialize workflow and benchmark based on dataset (DSL only)
        workflow = DslWorkflowRunner(
            name=f"dsl_workflow_config_{config_idx}",
            llm_configs=llm_configs,
            workflow_type=workflow_type_name,
            output_dir=config_dir,
            structure_id=structure_id,
        )

        file_path = resolve_existing_path(data_path) if data_path else None
        if not file_path:
            file_path = _default_data_path(dataset, split)
        if not file_path:
            raise ValueError(
                f"No data_path provided and benchmark '{benchmark_info['canonical_dataset_name']}' "
                f"has no DEFAULT_SPLIT_PATHS entry for split='{split}'."
            )
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        benchmark_kwargs = dict(benchmark_info.get("default_init_kwargs", {}) or {})
        workflow_type_name_norm = str(workflow_type_name or "").lower()
        if workflow_type_name_norm == "livecodebench" and entry_point_file:
            benchmark_kwargs["entry_point_file"] = entry_point_file
        for key, value in list(benchmark_kwargs.items()):
            if isinstance(value, str) and key.endswith("_file"):
                benchmark_kwargs[key] = resolve_existing_path(value) or value
        benchmark = get_benchmark(
            dataset,
            name=benchmark_info["canonical_dataset_name"],
            file_path=file_path,
            log_path=str(config_dir),
            **benchmark_kwargs,
        )

        all_data = await benchmark.load_data()
        logger.info(f"Evaluating all {len(all_data)} problems")

        # Run evaluation with full dataset
        logger.info(f"Starting evaluation for configuration {config_idx + 1}...")
        logger.info(f"Using max_tasks: {max_tasks}")
        results = await benchmark.evaluate_all_problems(all_data, workflow, max_concurrent_tasks=max_tasks)
        
        # Calculate scores manually since we're not using run_baseline
        columns = benchmark.get_result_columns()
        average_score = benchmark.save_results_to_csv(results, columns)

        # Annotate trace with unified score/metric fields
        annotate_trace_file_with_scores(workflow.trace_file, dataset, results)

        # Calculate success rate from results
        metric_name = _metric_for_dataset(dataset)
        scores = [_result_score(dataset, r) for r in results]
        success_count = sum(1 for s in scores if _score_is_success(metric_name, float(s)))
        total_problems = len(scores)
        success_rate = success_count / total_problems if total_problems > 0 else 0.0
        
        logger.info(f"Configuration {config_idx + 1} Results:")
        metric_name = benchmark_info["metric_name"]
        logger.info(f"  {metric_name}: {average_score:.4f}")
        logger.info(f"  Success Rate: {success_rate:.4f}")
        
        # Update configuration with actual results
        updated_config = config.copy()
        _attach_score_fields(updated_config, workflow_type_name, average_score)
        updated_config['actual_success_rate'] = success_rate
        updated_config['actual_score'] = average_score
        updated_config['actual_metric'] = metric_name
        updated_config['evaluation_timestamp'] = datetime.now().isoformat()
        updated_config['config_index'] = config_idx
        updated_config['output_dir'] = str(config_dir)
        
        # Save individual config results
        config_results_file = config_dir / "config_results.json"
        with open(config_results_file, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f, indent=2)
        
        # Update config info file with results
        config_info_file = config_dir / "config_info.json"
        if config_info_file.exists():
            with open(config_info_file, 'r', encoding='utf-8') as f:
                config_info = json.load(f)
            config_info['evaluation_end_time'] = datetime.now().isoformat()
            results_dict = {
                'actual_success_rate': success_rate,
                'total_problems': total_problems,
                'success_count': success_count
            }
            _attach_score_fields(results_dict, workflow_type_name, average_score)
            config_info['results'] = results_dict
            with open(config_info_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2)
        
        logger.info(f"Configuration {config_idx + 1} completed successfully!")
        
        return updated_config
        
    except Exception as e:
        logger.error(f"Error evaluating configuration {config_idx + 1}: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        traceback.print_exc()
        
        # Update config info file with error
        config_info_file = config_dir / "config_info.json"
        if config_info_file.exists():
            with open(config_info_file, 'r', encoding='utf-8') as f:
                config_info = json.load(f)
            config_info['evaluation_end_time'] = datetime.now().isoformat()
            config_info['evaluation_error'] = str(e)
            config_info['error_traceback'] = error_traceback
            with open(config_info_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2)
        
        # Return config with error status
        updated_config = config.copy()
        _attach_score_fields(updated_config, workflow_type_name, 0.0)
        updated_config['actual_success_rate'] = 0.0
        updated_config['evaluation_error'] = str(e)
        updated_config['evaluation_timestamp'] = datetime.now().isoformat()
        updated_config['config_index'] = config_idx
        updated_config['output_dir'] = str(config_dir)
        
        return updated_config
    finally:
        # Ensure we close any async LLM clients to avoid leaking connections across configs
        close_method = getattr(workflow, "llm", None)
        if close_method and hasattr(close_method, "aclose"):
            try:
                await close_method.aclose()
            except Exception:
                pass



async def run_validation(args):
    """Run workflow evaluations using CLI-provided args."""
    experiment_id = args.experiment_id
    benchmark_info_main = _resolve_benchmark(args.dataset)
    if args.config_file is None:
        args.config_file = f"results/{experiment_id}/compiled/compiled_configs.json"

    config_file = Path(args.config_file)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return 1

    with open(config_file, "r", encoding="utf-8") as f:
        workflow_data = json.load(f)

    if isinstance(workflow_data, dict) and workflow_data.get("schema_version") == "flowcompile.compiled.v1":
        logger.error(
            "flowcompile.compiled.v1 is no longer supported. "
            "Recompile with `flowcompile predict` to produce flowcompile.compiled.v2."
        )
        return 1
    elif isinstance(workflow_data, dict) and "levels" in workflow_data:
        logger.error(
            "Level-based config files are no longer supported. "
            "Use flat flowcompile.compiled.v2 configs."
        )
        return 1
    elif isinstance(workflow_data, dict) and any(key.startswith("level_") for key in workflow_data):
        logger.error(
            "Level-based config files are no longer supported. "
            "Use flat flowcompile.compiled.v2 configs."
        )
        return 1

    # Normalize to flat config pools.
    source_metadata = workflow_data.get("metadata", {}) if isinstance(workflow_data, dict) else {}
    source_data_for_split = {
        "pareto_optimal": [],
        "non_pareto_sample": [],
        "metadata": source_metadata,
    }

    pareto_sample_n = getattr(args, "pareto_sample_n", None)

    if isinstance(workflow_data, dict) and workflow_data.get("schema_version") == "flowcompile.compiled.v2":
        base_configs = list(workflow_data.get("configs", []))
        source_data_for_split["pareto_optimal"] = [c for c in base_configs if _is_pareto_config(c)]
    elif isinstance(workflow_data, dict) and isinstance(workflow_data.get("all"), list):
        base_configs = list(workflow_data.get("all", []))
        source_data_for_split["pareto_optimal"] = [c for c in base_configs if _is_pareto_config(c)]
    elif isinstance(workflow_data, dict):
        source_data_for_split["pareto_optimal"] = list(workflow_data.get("pareto_optimal", []))
        if not source_data_for_split["pareto_optimal"] and isinstance(workflow_data.get("all"), list):
            source_data_for_split["pareto_optimal"] = [
                c for c in list(workflow_data.get("all", [])) if _is_pareto_config(c)
            ]
    elif isinstance(workflow_data, list):
        source_data_for_split["pareto_optimal"] = [c for c in workflow_data if _is_pareto_config(c)]
    else:
        logger.error("Unsupported configuration file format.")
        return 1

    all_configs = list(source_data_for_split["pareto_optimal"])
    if not all_configs:
        logger.error(
            "No Pareto configurations found in input. "
            "Recompile with `flowcompile predict` and verify Pareto outputs."
        )
        return 1

    logger.info(f"Evaluating {len(all_configs)} Pareto optimal configurations")

    sampling_metadata: Optional[Dict[str, Any]] = None
    if pareto_sample_n is not None:
        all_configs, sampling_metadata = _sample_pareto_even_by_latency(all_configs, pareto_sample_n)
        logger.info(
            "Selected Pareto sample: "
            f"{sampling_metadata['selected_count']}/{sampling_metadata['candidate_count']} "
            f"(n={sampling_metadata['requested_n']}, "
            f"latency={sampling_metadata['latency_min']:.4f}-{sampling_metadata['latency_max']:.4f}s)"
        )

    evaluation_items = _build_evaluation_items(all_configs)

    random.seed(args.random_seed)
    random.shuffle(evaluation_items)
    logger.info(f"Shuffled configuration order for evaluation (seed: {args.random_seed})")

    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = Path(f"results/{experiment_id}/evaluations/eval_{timestamp}")

    output_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")

    original_config_file = output_base_dir / "original_config.json"
    config_save_data = workflow_data.copy() if isinstance(workflow_data, dict) else {"all": workflow_data}
    if config_file:
        config_save_data["source_file"] = str(config_file)
    else:
        config_save_data["source_file"] = "generated_on_the_fly"
    with open(original_config_file, "w", encoding="utf-8") as f:
        json.dump(config_save_data, f, indent=2)

    updated_configs = []
    parallel_workers = args.parallel
    evaluated_count = 0
    skipped_count = 0
    total_items = len(evaluation_items)

    logger.info(f"\nEvaluating configurations with parallelism: {parallel_workers}")

    if parallel_workers == 1:
        for order_idx, item in enumerate(evaluation_items):
            config_idx, config = item
            if order_idx < args.start_idx:
                logger.info(f"Skipping configuration {order_idx + 1} (before start_idx={args.start_idx})")
                continue
            if args.end_idx is not None and order_idx >= args.end_idx:
                logger.info(f"Reached end_idx={args.end_idx}, stopping evaluation")
                break

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Progress: {order_idx + 1}/{total_items} (config_{config_idx:04d})")
            logger.info(f"{'=' * 80}")

            expected_config_id = config.get("_compiled_config_id") or config.get("config_id")
            existing_results = _load_existing_results(
                config_idx,
                output_base_dir,
                expected_config_id=expected_config_id,
            )
            if existing_results:
                logger.info(f"Configuration {order_idx + 1} (config_{config_idx:04d}) already evaluated, skipping...")
                updated_config = existing_results
                skipped_count += 1
            else:
                updated_config = await evaluate_configuration(
                    config,
                    config_idx,
                    output_base_dir,
                    args.workflow_type,
                    args.split,
                    args.dataset,
                    args.data_path,
                    getattr(args, "entry_point_file", None),
                    args.max_tasks,
                )
                evaluated_count += 1

            updated_configs.append(updated_config)

            _save_interim_results(
                updated_configs,
                source_data_for_split,
                output_base_dir,
                args.workflow_type,
                sampling_metadata=sampling_metadata,
            )
    else:
        semaphore = asyncio.Semaphore(parallel_workers)

        async def evaluate_with_semaphore(config, config_idx, order_idx):
            async with semaphore:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Starting Configuration {order_idx + 1}/{total_items} (config_{config_idx:04d})")
                logger.info(f"{'=' * 80}")

                expected_config_id = config.get("_compiled_config_id") or config.get("config_id")
                existing_results = _load_existing_results(
                    config_idx,
                    output_base_dir,
                    expected_config_id=expected_config_id,
                )
                if existing_results:
                    logger.info(f"Configuration {order_idx + 1} (config_{config_idx:04d}) already evaluated, skipping...")
                    return existing_results, True

                result = await evaluate_configuration(
                    config,
                    config_idx,
                    output_base_dir,
                    args.workflow_type,
                    args.split,
                    args.dataset,
                    args.data_path,
                    getattr(args, "entry_point_file", None),
                    args.max_tasks,
                )
                return result, False

        configs_to_evaluate = []
        for order_idx, item in enumerate(evaluation_items):
            config_idx, config = item
            if order_idx < args.start_idx:
                logger.info(f"Skipping configuration {order_idx + 1} (before start_idx={args.start_idx})")
                continue
            if args.end_idx is not None and order_idx >= args.end_idx:
                logger.info(f"Stopping at end_idx={args.end_idx}, not including config {order_idx + 1}")
                break
            configs_to_evaluate.append((config, config_idx, order_idx))

        logger.info(
            f"Evaluating {len(configs_to_evaluate)} configs in parallel "
            f"(from index {args.start_idx} to {args.end_idx if args.end_idx else total_items})"
        )

        tasks = [evaluate_with_semaphore(config, config_idx, order_idx) for config, config_idx, order_idx in configs_to_evaluate]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            updated_config, was_skipped = await coro
            updated_configs.append(updated_config)
            if was_skipped:
                skipped_count += 1
            else:
                evaluated_count += 1
            completed += 1

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Overall Progress: {completed}/{len(configs_to_evaluate)} completed")
            logger.info(f"{'=' * 80}")

            sorted_configs = sorted(updated_configs, key=lambda x: x.get("config_index", 0))
            _save_interim_results(
                sorted_configs,
                source_data_for_split,
                output_base_dir,
                args.workflow_type,
                sampling_metadata=sampling_metadata,
            )

        updated_configs = sorted(updated_configs, key=lambda x: x.get("config_index", 0))

    final_results = {
        "pareto_optimal": [],
        "non_pareto_sample": [],
        "metadata": source_data_for_split.get("metadata", {}),
        "evaluation_metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "workflow_type": args.workflow_type,
            "total_evaluated": len(updated_configs),
            "configurations_evaluated": evaluated_count,
            "configurations_skipped": skipped_count,
            "output_directory": str(output_base_dir),
            "config_file": str(config_file),
            "split": args.split,
        },
    }

    final_results["pareto_optimal"] = updated_configs
    final_results["non_pareto_sample"] = []
    if sampling_metadata is not None:
        final_results["evaluation_metadata"]["sampling"] = sampling_metadata

    accuracies = [c["actual_accuracy"] for c in updated_configs if "actual_accuracy" in c]
    if accuracies:
        final_results["evaluation_metadata"]["aggregate_statistics"] = {
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
        }

    final_file = output_base_dir / "workflow_results_final.json"
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 80)
    print("WORKFLOW EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total configurations processed: {len(updated_configs)}")
    print(f"Configurations evaluated: {evaluated_count}")
    print(f"Configurations skipped: {skipped_count}")
    print(f"Output directory: {output_base_dir}")
    print(f"Final results file: {final_file}")

    if accuracies:
        print("\nAggregate Statistics:")
        metric_name = benchmark_info_main["metric_name"]
        label = {
            "accuracy": "Accuracy",
            "f1": "F1 Score",
            "pass_at_1": "Pass Rate",
        }.get(metric_name, metric_name)
        print(f"  Mean {label}: {final_results['evaluation_metadata']['aggregate_statistics']['mean_accuracy']:.4f}")
        print(
            f"  {label} Range: "
            f"[{final_results['evaluation_metadata']['aggregate_statistics']['min_accuracy']:.4f}, "
            f"{final_results['evaluation_metadata']['aggregate_statistics']['max_accuracy']:.4f}]"
        )

    print("=" * 80)

    if parallel_workers > 1:
        print("\nParallel Execution Summary:")
        print(f"  Configurations processed: {len(updated_configs)}")
        print(f"  Configurations evaluated: {evaluated_count}")
        print(f"  Configurations skipped: {skipped_count}")
        print(f"  Parallel workers: {parallel_workers}")
        print(f"  Speedup: ~{min(parallel_workers, evaluated_count)}x (theoretical)")

    return 0
