"""
Utility functions for router data preparation and consolidation.

Provides functions to:
- Consolidate validation data from raw profiling results
- Load and prepare training data for routers
- Convert between different data formats
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from workflow_compiler.core.llm.config import parse_config

logger = logging.getLogger(__name__)


def row_to_runtime_config(
    row: pd.Series,
    workflow_type: str,
    config_id: str,
) -> Dict[str, Any]:
    """Convert a workflow config row into the runtime config schema."""
    agents: Dict[str, Any] = {}

    for col, value in row.items():
        if not col.endswith("_setting"):
            continue
        raw_agent = col[:-len("_setting")]
        if value is None or pd.isna(value):
            continue
        setting = value if isinstance(value, str) else str(value)
        model, budget = parse_config(setting)
        agents[raw_agent] = {
            "setting": setting,
            "model": model,
            "budget": budget,
        }

    return {
        "config_id": config_id,
        "workflow_type": workflow_type,
        "structure_id": row.get("structure_id"),
        "agents": agents,
        "metrics": {
            "expected_accuracy": float(row.get("workflow_accuracy", 0.0)),
            "expected_latency": float(row.get("workflow_latency", 0.0)),
        },
        "pareto": {
            "is_pareto": bool(row.get("is_pareto", False)),
            "rank": int(row.get("pareto_rank", 0)),
        },
    }


def consolidate_validation_data(
    detailed_results_files: Union[str, List[str]],
    trace_data_file: str,
    latency_file: str,
    workflow_type: str,
    model_to_hf_name: Optional[Dict[str, str]] = None,
    data_files: Optional[Union[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Consolidate validation set data from raw profiling data into a query-indexed structure.
    
    Args:
        detailed_results_files: Path(s) to detailed_results.json file(s)
        trace_data_file: Path to trace_training_data.json
        latency_file: Path to latency benchmark file
        workflow_type: 'math', 'hotpotqa', or 'livecodebench'
        model_to_hf_name: Optional mapping of model names to HuggingFace names
        data_files: Optional path(s) to data files (e.g., test.jsonl, validate.jsonl) to lookup unique_ids
    
    Returns:
        query_data_table: Dict mapping query_id -> {
            'query_text': str,
            'agents': {
                'agent_name': {
                    'setting_name': {
                        'accuracy': float (0 or 1),
                        'latency': float
                    }
                }
            }
        }
    """
    logger.info("="*80)
    logger.info("CONSOLIDATING VALIDATION DATA FROM RAW PROFILING DATA")
    logger.info("="*80)
    
    # Default model mapping if not provided
    if model_to_hf_name is None:
        from workflow_compiler.core.analysis import MODEL_TO_HF_NAME as default_mapping
        model_to_hf_name = default_mapping
    
    # Load and merge detailed_results from multiple files
    detailed_results = {}
    if isinstance(detailed_results_files, str):
        detailed_results_files = [detailed_results_files]
    
    logger.info(f"Loading {len(detailed_results_files)} detailed_results file(s)...")
    for filename in detailed_results_files:
        logger.info(f"  Loading {filename}...")
        with open(filename, 'r') as f:
            file_data = json.load(f)
            for subagent, settings in file_data.items():
                if subagent not in detailed_results:
                    detailed_results[subagent] = {}
                for setting, entries in settings.items():
                    if setting not in detailed_results[subagent]:
                        detailed_results[subagent][setting] = []
                    detailed_results[subagent][setting].extend(entries)
    
    logger.info(f"Merged data contains {len(detailed_results)} subagent(s)")
    
    # Load trace training data
    logger.info(f"Loading trace training data from: {trace_data_file}")
    with open(trace_data_file, 'r') as f:
        trace_training_data = json.load(f)
    
    # Load latency benchmark data
    logger.info(f"Loading latency benchmark data from: {latency_file}")
    with open(latency_file, 'r') as f:
        latency_data = json.load(f)
    
    # Build model to latency mapping
    model_to_io_latency_per_token = {}
    for model_name, model_data in latency_data.items():
        model_data = model_data[0]
        prefill_throughput = model_data["prefill_tok_per_s"]
        decode_throughput = model_data["decode_tok_per_s"]
        prefill_latency_per_token = 1.0 / prefill_throughput
        decode_latency_per_token = 1.0 / decode_throughput
        model_to_io_latency_per_token[model_name] = {
            "prefill_latency_per_token": prefill_latency_per_token,
            "decode_latency_per_token": decode_latency_per_token,
        }
    
    # Load unique_id mapping and full metadata from data files if provided
    problem_to_unique_id = {}
    problem_to_full_data = {}  # Store full JSONL data for enrichment
    if data_files:
        logger.info("Loading unique_id mapping from data files...")
        if isinstance(data_files, str):
            data_files = [data_files]
        
        for data_file in data_files:
            logger.info(f"  Loading {data_file}...")
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    # For livecodebench, the problem text is in 'question_content'
                    # For other workflows, it's in 'problem'
                    problem_text = item.get('problem') or item.get('question_content')
                    unique_id = item.get('unique_id')
                    if problem_text and unique_id:
                        problem_to_unique_id[problem_text] = unique_id
                    # Also store the full item for metadata enrichment
                    if problem_text:
                        problem_to_full_data[problem_text] = item
        
        logger.info(f"Loaded {len(problem_to_unique_id)} unique_id mappings and {len(problem_to_full_data)} full data entries")
    
    # Create problem to metadata mapping
    logger.info("Creating problem to metadata mapping...")
    problem_to_metadata = {}
    for training_data in trace_training_data["training_data"]:
        original_sample = training_data["original_sample"]
        problem = training_data.get("problem")
        if not problem:
            problem = original_sample.get("problem")
        if problem:
            # Start with original_sample
            metadata = original_sample.copy()
            
            # Enrich with data from JSONL files if available
            if problem in problem_to_full_data:
                full_data = problem_to_full_data[problem]
                # Merge full data, keeping original_sample fields if they conflict
                for key, value in full_data.items():
                    if key not in metadata:
                        metadata[key] = value
            
            # Enrich metadata with unique_id if available
            if problem in problem_to_unique_id:
                metadata['unique_id'] = problem_to_unique_id[problem]
            problem_to_metadata[problem] = metadata
    
    # Build query-indexed data structure
    query_data_table = {}
    
    logger.info(f"Processing records by query...")
    total_entries = sum(len(entries) for subagent_data in detailed_results.values() 
                       for entries in subagent_data.values())
    
    with tqdm(total=total_entries, desc="Processing entries") as pbar:
        for subagent in detailed_results:
            for setting in detailed_results[subagent]:
                for entry in detailed_results[subagent][setting]:
                    problem = entry["problem"]
                    metadata = problem_to_metadata.get(problem)
                    
                    if metadata is None:
                        pbar.update(1)
                        continue
                    
                    # Determine query ID and text based on workflow type
                    if workflow_type == 'math':
                        query_id = metadata["unique_id"]
                        query_text = problem
                    elif workflow_type == 'hotpotqa':
                        query_id = metadata["_id"]
                        question = metadata["question"]
                        query_text = question
                    elif workflow_type == 'livecodebench':
                        query_id = metadata["question_id"]
                        query_text = metadata["question_content"]
                    else:
                        raise ValueError(f"Unknown workflow type: {workflow_type}")
                    
                    # Get accuracy and tokens
                    accuracy = entry["accuracy"]
                    input_tokens = entry["avg_input_tokens"]
                    output_tokens = entry["avg_output_tokens"]
                    
                    # Calculate latency
                    setting_parts = setting.split('_')
                    if len(setting_parts) >= 2:
                        budget_idx = setting_parts.index('budget') if 'budget' in setting_parts else len(setting_parts)
                        model_base = '_'.join(setting_parts[:budget_idx])
                    else:
                        model_base = setting_parts[0]
                    
                    hf_model_name = model_to_hf_name.get(model_base)
                    
                    if hf_model_name and hf_model_name in model_to_io_latency_per_token:
                        io_latency = model_to_io_latency_per_token[hf_model_name]
                        prefill_latency = input_tokens * io_latency['prefill_latency_per_token']
                        decode_latency = output_tokens * io_latency['decode_latency_per_token']
                        latency = prefill_latency + decode_latency
                    else:
                        latency = 0.0
                    
                    # Initialize query entry if not exists
                    if query_id not in query_data_table:
                        query_data_table[query_id] = {
                            'query_text': query_text,
                            'agents': {}
                        }
                    
                    # Initialize agent entry if not exists
                    if subagent not in query_data_table[query_id]['agents']:
                        query_data_table[query_id]['agents'][subagent] = {}
                    
                    # Store setting data
                    query_data_table[query_id]['agents'][subagent][setting] = {
                        'accuracy': accuracy,
                        'latency': latency
                    }
                    
                    pbar.update(1)
    
    logger.info(f"✓ Consolidated data for {len(query_data_table)} unique queries")
    
    # Print statistics
    logger.info("Data statistics:")
    logger.info(f"  Total queries: {len(query_data_table)}")
    
    # Sample a query to show structure
    if query_data_table:
        sample_query_id = list(query_data_table.keys())[0]
        sample_data = query_data_table[sample_query_id]
        logger.info(f"Sample query structure (query_id: {sample_query_id[:50]}...):")
        logger.info(f"  Agents: {list(sample_data['agents'].keys())}")
        for agent in list(sample_data['agents'].keys())[:2]:
            n_settings = len(sample_data['agents'][agent])
            logger.info(f"    {agent}: {n_settings} settings")
    
    return query_data_table


def load_test_queries(
    test_data_file: str,
    workflow_type: str
) -> tuple[List[Dict[str, Any]], str]:
    """
    Load test queries from benchmark file.
    
    Args:
        test_data_file: Path to test data JSONL file
        workflow_type: 'math', 'hotpotqa', 'livecodebench', or 'gsm8k'
    
    Returns:
        Tuple of (test_queries list, id_key_name string)
        test_queries: List of dicts with 'query_id' and 'query_text'
        id_key_name: The key name used for the problem ID
    """
    logger.info("="*80)
    logger.info("LOADING TEST QUERIES")
    logger.info("="*80)
    logger.info(f"Loading from: {test_data_file}")
    
    test_queries = []
    id_key_name = None
    
    with open(test_data_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Detect ID key on first iteration
            if id_key_name is None:
                if workflow_type in ('math', 'gsm8k'):
                    id_key_name = 'unique_id'
                elif workflow_type == 'hotpotqa':
                    id_key_name = '_id'
                elif workflow_type == 'livecodebench':
                    id_key_name = 'question_id'
                logger.info(f"Detected ID key: '{id_key_name}'")
            
            if workflow_type in ('math', 'gsm8k'):
                query_id = data.get("unique_id", str(len(test_queries)))
                query_text = data.get('problem', '')
            elif workflow_type == 'hotpotqa':
                query_id = data.get('_id', str(len(test_queries)))
                question = data.get('question', '')
                query_text = question
            elif workflow_type == 'livecodebench':
                query_id = data.get('question_id', str(len(test_queries)))
                query_text = data.get('question_content', data.get('problem', ''))
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            test_queries.append({
                'query_id': query_id,
                'query_text': query_text
            })
    
    logger.info(f"✓ Loaded {len(test_queries)} test queries")
    logger.info(f"ID key name: {id_key_name}")
    
    return test_queries, id_key_name


def convert_query_table_to_training_format(
    query_data_table: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convert query data table to router training format.
    
    Args:
        query_data_table: Dict mapping query_id -> query data
    
    Returns:
        List of training examples in router format
    """
    training_data = []
    
    for query_id, query_data in query_data_table.items():
        training_data.append({
            'query_id': query_id,
            'query_text': query_data['query_text'],
            'agents': query_data['agents']
        })
    
    return training_data


def save_consolidated_data(
    query_data_table: Dict[str, Any],
    output_file: str
):
    """
    Save consolidated data to JSON file.
    
    Args:
        query_data_table: Query data table
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(query_data_table, f, indent=2)
    
    logger.info(f"✓ Saved consolidated data to: {output_file}")


def load_consolidated_data(input_file: str) -> Dict[str, Any]:
    """
    Load consolidated data from JSON file.
    
    Args:
        input_file: Input file path
    
    Returns:
        Query data table
    """
    logger.info(f"Loading consolidated data from: {input_file}")
    
    with open(input_file, 'r') as f:
        query_data_table = json.load(f)
    
    logger.info(f"✓ Loaded data for {len(query_data_table)} queries")
    
    return query_data_table
