#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate KNN configurations from query-based JSON file.
For each query's configurations, evaluates them in parallel and saves actual metrics
including per-agent token counts back to the same JSON structure.

Metadata is automatically extracted from the JSON file (test_data_file, dataset).
Benchmark is initialized once and reused across all query configurations.
"""

import json
import asyncio
import random
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import threading
from collections import defaultdict
from tqdm import tqdm

# Import the evaluation components
from workflow_compiler.dsl.runtime import DslWorkflowRunner
from workflow_compiler.benchmarks import get_benchmark, get_benchmark_info
from workflow_compiler.core.logs import logger


def _resolve_benchmark(dataset: str) -> Dict[str, Any]:
    info = get_benchmark_info(dataset)
    benchmark_class = info["class"]
    info["canonical_dataset_name"] = info["name"]
    info["workflow_type"] = info.get("workflow_type") or getattr(benchmark_class, "WORKFLOW_TYPE", "math")
    info["metric_name"] = info.get("metric_name") or getattr(benchmark_class, "METRIC_NAME", "accuracy")
    info["default_init_kwargs"] = dict(
        info.get("default_init_kwargs") or getattr(benchmark_class, "DEFAULT_INIT_KWARGS", {}) or {}
    )
    return info


def _score_from_result(dataset: str, result: Any) -> float:
    benchmark_class = _resolve_benchmark(dataset)["class"]
    if hasattr(benchmark_class, "score_from_result"):
        return float(benchmark_class.score_from_result(result))
    return float(result[-1])


def _build_llm_configs_for_workflow(workflow_type: str, subagent_settings: Dict[str, Any]) -> Dict[str, Any]:
    if workflow_type in {"math", "gsm8k"}:
        return {
            "meta": None,
            "programmer": subagent_settings.get("programmer"),
            "sc_ensemble": subagent_settings.get("sc_ensemble"),
            "refine_solver": subagent_settings.get("refine"),
            "detailed_solver": subagent_settings.get("detailed"),
            "generate_solver": subagent_settings.get("generate1") or subagent_settings.get("generate"),
        }
    if workflow_type == "hotpotqa":
        sc_ensemble = subagent_settings.get("sc_ensemble")
        return {
            "meta": sc_ensemble,
            "answer_generate": subagent_settings.get("answer_generate"),
            "sc_ensemble": sc_ensemble,
            "format_answer": subagent_settings.get("format_answer"),
        }
    if workflow_type == "livecodebench":
        sc_ensemble = subagent_settings.get("sc_ensemble")
        reflection_test = subagent_settings.get("reflection_test")
        test_setting = subagent_settings.get("test") or reflection_test
        return {
            "meta": sc_ensemble,
            "code_generate": subagent_settings.get("code_generate"),
            "sc_ensemble": sc_ensemble,
            "test": test_setting,
            "reflection_test": reflection_test,
        }
    raise ValueError(f"Unsupported workflow_type: {workflow_type}")


def _score_is_success(metric: str, score: float) -> bool:
    if metric == "f1":
        return score > 0.5
    return score >= 0.5

async def evaluate_single_configuration(
    config: Dict[str, Any],
    query_id: str,
    config_idx: int,
    output_base_dir: Path,
    benchmark: Any,
    benchmark_info: Dict[str, Any],
    filtered_data: List[Dict[str, Any]],
    workflow: Any,
    dataset: str = "MATH"
) -> Dict[str, Any]:
    """
    Evaluate a single workflow configuration for a specific query.
    
    Args:
        config: Configuration dictionary with agent settings
        query_id: The query/problem ID (e.g., "test/precalculus/1303.json")
        config_idx: Index of configuration in the query's config list
        output_base_dir: Base directory for all outputs
        benchmark: Already initialized benchmark instance
        filtered_data: Pre-filtered problem data for this query
        workflow: Workflow instance already configured for this config
        dataset: Dataset name ('MATH', 'GSM8K', 'HotpotQA', or 'LiveCodeBench')
    
    Returns:
        Updated configuration with actual metrics and per-agent token counts
    """
    # Create output directory for this specific config
    safe_query_id = query_id.replace('/', '_').replace('.json', '')
    config_dir = output_base_dir / safe_query_id / f"config_{config_idx:04d}"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    structure_id = config.get('structure_id', None)
    subagent_settings = config.get('subagent_settings', {})
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating Query: {query_id}, Config {config_idx + 1}")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Structure: {structure_id}")
    logger.info(f"Subagent settings: {subagent_settings}")
    logger.info(f"Output directory: {config_dir}")
    
    try:
        logger.info(f"Starting evaluation for {query_id}...")
        
        # Run evaluation directly since we only have 1 problem (no need for async overhead)
        results = [await benchmark.evaluate_problem(filtered_data[0], workflow)]
        
        # Extract metrics from results
        # Different benchmarks return different tuple structures:
        # - MATH/GSM8K: (question, prediction, expected_output, score) - 4 items
        # - HotpotQA: (question, context, prediction, expected_output, score) - 5 items
        # - LiveCodeBench: (question, prediction, expected_output, score, evaluation_details) - 5 items
        result_tuple = results[0]

        average_score = _score_from_result(dataset, result_tuple)
        evaluation_details: Optional[Dict[str, Any]] = None
        if isinstance(result_tuple, tuple) and result_tuple and isinstance(result_tuple[-1], dict):
            evaluation_details = result_tuple[-1]

        # Extract per-agent token usage from trace file metadata
        agent_token_usage = {}
        
        # Read trace file to get per-step token usage
        if workflow.trace_file.exists():
            with open(workflow.trace_file, 'r', encoding='utf-8') as f:
                trace_content = f.read().strip()
                if trace_content:
                    # Trace file is JSONL format - one JSON object per line
                    for line in trace_content.split('\n'):
                        if not line.strip():
                            continue
                        trace_entry = json.loads(line)
                        
                        # Each trace entry has a 'steps' list with metadata for each agent call
                        for step in trace_entry.get('steps', []):
                            agent_name = step.get('agent', '')
                            metadata = step.get('metadata', {})
                            
                            # Initialize agent stats if not exists
                            if agent_name and agent_name not in agent_token_usage:
                                agent_token_usage[agent_name] = {
                                    'input_tokens': 0,
                                    'output_tokens': 0,
                                    'total_tokens': 0,
                                }
                            
                            # Accumulate token usage from metadata
                            if agent_name:
                                agent_token_usage[agent_name]['input_tokens'] += metadata.get('input_tokens', 0)
                                agent_token_usage[agent_name]['output_tokens'] += metadata.get('output_tokens', 0)
                                agent_token_usage[agent_name]['total_tokens'] += (
                                    metadata.get('input_tokens', 0) + metadata.get('output_tokens', 0)
                                )

        # Annotate trace entries with unified score/metric
        if workflow.trace_file.exists():
            try:
                trace_entries = []
                with open(workflow.trace_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            trace_entries.append(json.loads(line))
                metric_name = benchmark_info["metric_name"]
                for entry in trace_entries:
                    # Remove legacy evaluation fields if present
                    for legacy_key in ["is_correct", "f1_score", "pass_at_1", "private_test_passed", "extracted_answer", "test_passed", "correct"]:
                        entry.pop(legacy_key, None)
                    entry["score"] = float(average_score)
                    entry["metric"] = metric_name
                tmp_file = workflow.trace_file.with_suffix(".jsonl.tmp")
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    for entry in trace_entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                tmp_file.replace(workflow.trace_file)
            except Exception as e:
                logger.warning(f"Failed to annotate trace with score/metric: {e}")
        
        metric = benchmark_info["metric_name"]
        is_correct = _score_is_success(metric, float(average_score))
        f1_score = None
        exact_match = None
        if metric == "f1":
            f1_score = average_score
        
        logger.info(f"Query {query_id}, Config {config_idx + 1} Results:")
        logger.info(f"  Correctness: {is_correct}")
        logger.info(f"  Score ({metric}): {average_score:.4f}")
        if exact_match is not None:
            logger.info(f"  Exact Match: {exact_match}")
        logger.info(f"  Average Score: {average_score:.4f}")
        if evaluation_details:
            logger.info(f"  Evaluation Details: {evaluation_details}")
        logger.info(f"  Agent Token Usage: {agent_token_usage}")
        
        # Update configuration with actual results
        updated_config = config.copy()
        updated_config['actual_correctness'] = is_correct
        updated_config['actual_score'] = average_score
        updated_config['actual_metric'] = metric
        updated_config['actual_accuracy'] = average_score
        if f1_score is not None:
            updated_config['actual_f1'] = f1_score
        if metric == "pass_at_1":
            updated_config['actual_pass_rate'] = average_score
        if exact_match is not None:
            updated_config['actual_exact_match'] = exact_match
        if evaluation_details:
            updated_config['evaluation_details'] = evaluation_details
        updated_config['agent_token_usage'] = agent_token_usage
        updated_config['evaluation_timestamp'] = datetime.now().isoformat()
        updated_config['config_index'] = config_idx
        updated_config['output_dir'] = str(config_dir)
        
        # Save individual config results
        config_results_file = config_dir / "config_results.json"
        with open(config_results_file, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f, indent=2)
        
        logger.info(f"Query {query_id}, Config {config_idx + 1} completed successfully!")
        
        return updated_config
        
    except Exception as e:
        logger.error(f"Error evaluating query {query_id}, config {config_idx + 1}: {e}")
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        
        # Return config with error status
        updated_config = config.copy()
        updated_config['actual_correctness'] = False
        updated_config['actual_score'] = 0.0
        updated_config['actual_metric'] = benchmark_info["metric_name"]
        updated_config['actual_accuracy'] = 0.0
        if benchmark_info["metric_name"] == "f1":
            updated_config['actual_f1'] = 0.0
        if benchmark_info["metric_name"] == "pass_at_1":
            updated_config['actual_pass_rate'] = 0.0
        updated_config['agent_token_usage'] = {}
        updated_config['evaluation_error'] = str(e)
        updated_config['evaluation_timestamp'] = datetime.now().isoformat()
        updated_config['config_index'] = config_idx
        updated_config['output_dir'] = str(config_dir)
        
        return updated_config


async def init_benchmark_for_dataset(
    benchmark_type: str,
    dataset_path: str,
    output_dir: Path
) -> Tuple[Any, Dict[str, Any]]:
    """
    Initialize benchmark instance for the given dataset type.
    
    Args:
        benchmark_type: Type of benchmark ('MATH', 'GSM8K', 'HotpotQA', 'LiveCodeBench')
        dataset_path: Path to the dataset file
        output_dir: Directory for outputs
    
    Returns:
        Tuple of (initialized benchmark instance, benchmark metadata)
    """
    benchmark_info = _resolve_benchmark(benchmark_type)
    benchmark_kwargs = dict(benchmark_info.get("default_init_kwargs", {}) or {})
    benchmark = get_benchmark(
        benchmark_type,
        name=benchmark_info["canonical_dataset_name"],
        file_path=dataset_path,
        log_path=str(output_dir),
        **benchmark_kwargs,
    )
    return benchmark, benchmark_info


async def run_knn_evaluate(args):
    """Evaluate all query configurations from a KNN routing JSON file."""
    if not getattr(args, "config_file", None):
        raise SystemExit("config_file is required for knn-evaluate")
    if getattr(args, "parallel", None) is None:
        args.parallel = 32
    if getattr(args, "random_seed", None) is None:
        args.random_seed = 42
    if getattr(args, "shuffle", None) is None:
        args.shuffle = False
    if getattr(args, "resume", None) is None:
        args.resume = False

    # Set random seed
    random.seed(args.random_seed)

    if getattr(args, "workflow_type", "dsl") not in ("dsl", "fixed", ""):
        logger.warning("KNN evaluation uses DSL workflows; workflow_type flag is ignored.")
    
    # Read configurations
    config_file = Path(args.config_file)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return 1
    
    logger.info(f"Loading configurations from: {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        workflow_data = json.load(f)
    
    # Extract metadata from the JSON file
    metadata = workflow_data.get('_metadata', {})
    test_data_file = metadata.get('test_data_file')
    dataset_type = metadata.get('dataset', 'MATH')
    workflow_type = metadata.get('workflow_type')
    id_key_name = metadata.get('id_key_name', 'id')  # Default to 'id' if not provided
    
    if not test_data_file:
        logger.error("Metadata '_metadata' with 'test_data_file' not found in config file")
        logger.error("Please run `flowcompile runtime knn` first to generate configs with metadata")
        return 1
    
    logger.info(f"Metadata extracted from config file:")
    logger.info(f"  Dataset: {dataset_type}")
    logger.info(f"  Test data file: {test_data_file}")
    if workflow_type:
        logger.info(f"  Workflow type (metadata): {workflow_type}")
    logger.info(f"  ID key name: {id_key_name}")
    
    # Set output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # Default: add _evaluated suffix before .json
        output_file = config_file.parent / f"{config_file.stem}_evaluated.json"
    
    # Check for resume
    if args.resume and output_file.exists():
        logger.info(f"Resuming from existing file: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
    
    # Set output directory for traces
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = config_file.parent / f"{config_file.stem}_traces"
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for traces: {output_base_dir}")
    
    # Get list of queries to evaluate (skip _metadata)
    query_items = [(k, v) for k, v in workflow_data.items() if k != '_metadata']
    total_queries = len(query_items)
    logger.info(f"Total queries to evaluate: {total_queries}")
    
    # Initialize benchmark once for all queries
    logger.info(f"\n{'='*80}")
    logger.info(f"Initializing benchmark for {dataset_type}")
    logger.info(f"Dataset path: {test_data_file}")
    logger.info(f"{'='*80}")
    
    benchmark, benchmark_info = await init_benchmark_for_dataset(
        benchmark_type=dataset_type,
        dataset_path=test_data_file,
        output_dir=output_base_dir
    )
    workflow_type = benchmark_info["workflow_type"]
    metric_name = benchmark_info["metric_name"]
    logger.info(f"✓ Benchmark initialized")
    logger.info(f"  Canonical benchmark: {benchmark_info['canonical_dataset_name']}")
    logger.info(f"  Resolved workflow type: {workflow_type}")
    logger.info(f"  Metric: {metric_name}")

    all_data = await benchmark.load_data()
    indexed_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for problem in all_data:
        problem_id = problem.get(id_key_name)
        if problem_id is not None:
            indexed_data[str(problem_id)].append(problem)

    # Flatten all query-config pairs into a single list for full parallelization
    all_tasks = []
    for query_idx, (query_id, query_data) in enumerate(query_items):
        configurations = query_data.get('configurations', [])
        for config_idx, config in enumerate(configurations):
            all_tasks.append({
                'query_idx': query_idx,
                'query_id': query_id,
                'query_data': query_data,
                'config_idx': config_idx,
                'config': config
            })
    
    total_tasks = len(all_tasks)
    logger.info(f"Total evaluations to run: {total_tasks} (across {total_queries} queries)")
    
    # Shuffle tasks if requested
    if args.shuffle:
        random.shuffle(all_tasks)
        logger.info(f"Shuffled evaluation order (seed: {args.random_seed})")
    
    # Parallelize all evaluations
    results_lock = threading.Lock()
    completed_count = 0
    semaphore = asyncio.Semaphore(args.parallel)
    progress_bar = tqdm(total=total_tasks, desc="Evaluating")
    
    async def evaluate_single_task(task):
        """Evaluate a single query-config pair with its own workflow instance."""
        nonlocal completed_count
        
        async with semaphore:
            query_id = task['query_id']
            query_data = task['query_data']
            config_idx = task['config_idx']
            config = task['config']
            
            # Load and filter data for this query
            filtered_data = indexed_data.get(str(query_id), [])
            
            if not filtered_data:
                logger.warning(f"No problem found with ID {query_id}")
                progress_bar.update(1)
                return None
            
            # Extract configuration for workflow initialization
            structure_id = config.get('structure_id', None)
            subagent_settings = config.get('subagent_settings', {})
            
            # Create output directory
            safe_query_id = query_id.replace('/', '_').replace('.json', '')
            config_dir = output_base_dir / safe_query_id / f"config_{config_idx:04d}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Build llm_configs for this specific configuration
            llm_configs = _build_llm_configs_for_workflow(workflow_type, subagent_settings)
            workflow = DslWorkflowRunner(
                name=f"workflow_{safe_query_id}_config_{config_idx}",
                llm_configs=llm_configs,
                workflow_type=workflow_type,
                output_dir=config_dir,
                structure_id=structure_id
            )
            
            # Evaluate this single config
            updated_config = await evaluate_single_configuration(
                config=config,
                query_id=query_id,
                config_idx=config_idx,
                output_base_dir=output_base_dir,
                benchmark=benchmark,
                benchmark_info=benchmark_info,
                filtered_data=filtered_data,
                workflow=workflow,
                dataset=dataset_type
            )
            
            # Thread-safe update of results
            with results_lock:
                # Initialize query data if not exists
                if query_id not in workflow_data or not isinstance(workflow_data[query_id], dict):
                    workflow_data[query_id] = {
                        'query': query_data.get('query', ''),
                        'top_k_similar_queries': query_data.get('top_k_similar_queries', []),
                        'configurations': []
                    }
                
                # Add or update config in the list
                if 'configurations' not in workflow_data[query_id]:
                    workflow_data[query_id]['configurations'] = []
                
                # Replace config at the correct position (by index)
                config_list = workflow_data[query_id]['configurations']
                # Extend list if needed to accommodate this config_idx
                while len(config_list) <= config_idx:
                    config_list.append(None)
                # Replace at the correct position (overwrites old unevaluated config)
                config_list[config_idx] = updated_config
                
                completed_count += 1
                
                # Save interim results periodically (every 10 completions)
                if completed_count % 10 == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(workflow_data, f, indent=2)
            
            progress_bar.update(1)
            return updated_config
    
    # Create all tasks and execute in parallel
    evaluation_tasks = [evaluate_single_task(task) for task in all_tasks]
    await asyncio.gather(*evaluation_tasks)
    progress_bar.close()
    
    # Calculate summary for each query
    for query_id, query_data in workflow_data.items():
        if query_id != '_metadata' and 'configurations' in query_data:
            configs = query_data['configurations']
            correct_count = sum(1 for c in configs if c.get('actual_correctness', False))
            
            query_data['evaluation_summary'] = {
                'total_configs': len(configs),
                'correct_count': correct_count,
                'accuracy': correct_count / len(configs) if configs else 0.0,
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    # Save final results
    logger.info(f"\nSaving final results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total queries evaluated: {total_queries}")
    print(f"Results saved to: {output_file}")
    print(f"Traces saved to: {output_base_dir}")
    
    # Calculate overall statistics
    total_configs = 0
    total_correct = 0
    
    for query_id, query_data in workflow_data.items():
        if query_id != '_metadata' and 'evaluation_summary' in query_data:
            summary = query_data['evaluation_summary']
            total_configs += summary.get('total_configs', 0)
            total_correct += summary.get('correct_count', 0)
    
    if total_configs > 0:
        print(f"\nOverall Statistics:")
        print(f"  Total configurations evaluated: {total_configs}")
        print(f"  Total correct: {total_correct}")
        print(f"  Overall accuracy: {total_correct / total_configs:.4f}")
    
    print("="*80)
    
    return 0
