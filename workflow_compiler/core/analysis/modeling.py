"""
Analysis utilities for FlowCompile.

Provides:
- Model name mapping and resolution
- Latency calculation
- Token counting
- Workflow accuracy computation
- Pareto frontier analysis
"""

import re
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Model Name Mapping
# ============================================================================

MODEL_TO_HF_NAME = {
    "qwen3-4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "deepseek-r1-qwen-1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "ds-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "qwen3-30b-a3b-thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen35-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen35-2b": "Qwen/Qwen3.5-2B",
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen35-9b": "Qwen/Qwen3.5-9B",
    "qwen35-27b": "Qwen/Qwen3.5-27B",
    "qwen35-0.8b-local": "Qwen/Qwen3.5-0.8B",
    "qwen35-2b-local": "Qwen/Qwen3.5-2B",
    "qwen35-4b-local": "Qwen/Qwen3.5-4B",
    "qwen35-9b-local": "Qwen/Qwen3.5-9B",
    "qwen35-27b-local": "Qwen/Qwen3.5-27B",
    "ministral-14b": "mistralai/Ministral-3-14B-Reasoning-2512",
    "qwq-32b": "Qwen/QwQ-32B",
    "ds-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}


def get_hf_model_name(model_name: str) -> str:
    """
    Get HuggingFace model name from short name.
    
    Args:
        model_name: Short model name (e.g., 'qwen3-4b')
    
    Returns:
        HuggingFace model name (e.g., 'Qwen/Qwen3-4B')
    """
    resolved = MODEL_TO_HF_NAME.get(model_name)
    if resolved is not None:
        return resolved
    # Keep local alias compatibility (e.g., qwen35-4b-local -> qwen35-4b).
    if model_name.endswith("-local"):
        stripped = model_name[:-len("-local")]
        resolved = MODEL_TO_HF_NAME.get(stripped)
        if resolved is not None:
            return resolved
    return model_name


def extract_model_name(
    setting: str,
    return_hf_name: bool = False,
    return_budget: bool = False
) -> Union[str, Tuple[str, int]]:
    """
    Extract model name and optionally budget from setting string.
    
    Args:
        setting: Setting string like 'qwen3-14b_budget_3000'
        return_hf_name: If True, return HuggingFace name; otherwise base name
        return_budget: If True, return tuple (model_name, budget); otherwise just model_name
    
    Returns:
        - If return_budget=False: Model name (str)
        - If return_budget=True: Tuple of (model_name, budget) where budget is int
    
    Examples:
        >>> extract_model_name('qwen3-4b_budget_1000')
        'qwen3-4b'
        >>> extract_model_name('qwen3-4b_budget_1000', return_budget=True)
        ('qwen3-4b', 1000)
        >>> extract_model_name('qwen3-4b_budget_1000', return_hf_name=True)
        'Qwen/Qwen3-4B'
    """
    # Handle non-string settings
    if not isinstance(setting, str):
        if return_budget:
            return str(setting), 0
        return str(setting)
    
    # Extract base model name before '_budget_'
    base_name = setting.split('_budget_')[0] if '_budget_' in setting else setting
    
    # Extract budget if requested
    if return_budget:
        budget = 0
        if '_budget_' in setting:
            try:
                budget_str = setting.split('_budget_')[1].split('_')[0].split('-')[0]
                if budget_str == 'unlimited':
                    budget = -1  # Special marker for unlimited
                else:
                    budget = int(budget_str)
            except (IndexError, ValueError):
                budget = 0
        
        # Return model name with budget
        if return_hf_name:
            return get_hf_model_name(base_name), budget
        return base_name, budget
    
    # Return just model name
    if return_hf_name:
        return get_hf_model_name(base_name)
    return base_name


# ============================================================================
# Latency Calculation
# ============================================================================

def load_latency_data(latency_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load latency benchmark data from JSON file.
    
    Args:
        latency_file: Path to JSON file with benchmark results
    
    Returns:
        Dictionary mapping HF model name to latency info:
        {
            "Qwen/Qwen3-4B": {
                "prefill_latency_per_token": 0.0002,
                "decode_latency_per_token": 0.002
            },
            ...
        }
    """
    import json
    
    try:
        with open(latency_file, 'r') as f:
            latency_data = json.load(f)
        
        model_to_io_latency_per_token = {}
        for model_name, model_data in latency_data.items():
            model_data = model_data[0] if isinstance(model_data, list) else model_data
            prefill_throughput = model_data["prefill_tok_per_s"]
            decode_throughput = model_data["decode_tok_per_s"]
            prefill_latency_per_token = 1.0 / prefill_throughput
            decode_latency_per_token = 1.0 / decode_throughput
            model_to_io_latency_per_token[model_name] = {
                "prefill_latency_per_token": prefill_latency_per_token,
                "decode_latency_per_token": decode_latency_per_token,
            }
        return model_to_io_latency_per_token
    except FileNotFoundError:
        raise FileNotFoundError(f"Latency file not found: {latency_file}")


def get_default_latency_data() -> Dict[str, Dict[str, float]]:
    """
    Get a conservative fallback latency table.

    Returns:
        Dictionary mapping HF model name to per-token latency values.
    """
    # Provide stable defaults for tests and fallback execution paths.
    # These values are placeholders and should be overridden by measured latency data.
    return {
        hf_name: {
            "prefill_latency_per_token": 0.0002,
            "decode_latency_per_token": 0.002,
        }
        for hf_name in MODEL_TO_HF_NAME.values()
    }

def calculate_latency(
    input_tokens: int,
    output_tokens: int,
    model_name: str,
    latency_data: Dict[str, Dict[str, float]]
) -> float:
    """
    Calculate latency for a given model and token counts.
    
    Args:
        input_tokens: Number of input tokens (prefill)
        output_tokens: Number of output tokens (decode)
        model_name: Model name or setting string (e.g., 'qwen3-4b_budget_1000')
        latency_data: Latency data dictionary from load_latency_data()
    
    Returns:
        Total latency in seconds
    """
    # Extract base model name and get HF name
    hf_model_name = extract_model_name(model_name, return_hf_name=True)
    
    if hf_model_name in latency_data:
        io_latency = latency_data[hf_model_name]
        prefill_latency = input_tokens * io_latency['prefill_latency_per_token']
        decode_latency = output_tokens * io_latency['decode_latency_per_token']
        return prefill_latency + decode_latency
    else:
        # Return 0 if model not found (caller should handle)
        return 0.0


def calculate_trace_latency(
    trace_data: Dict,
    latency_data: Dict[str, Dict[str, float]],
    llm_configs: Optional[Dict[str, str]] = None
) -> float:
    """
    Calculate total latency for a trace (all steps).
    
    Args:
        trace_data: Single trace dictionary with 'steps' key
        latency_data: Latency data dictionary
        llm_configs: Optional mapping of agent names to model settings
    
    Returns:
        Total latency in seconds
    """
    total_latency = 0.0
    
    if 'steps' not in trace_data:
        return 0.0
    
    for step in trace_data['steps']:
        if 'metadata' not in step:
            continue
        
        metadata = step['metadata']
        input_tokens = metadata.get('input_tokens', 0)
        output_tokens = metadata.get('output_tokens', 0)
        
        # Determine model for this step
        agent_name = step.get('agent', '')
        model_name = None
        
        if llm_configs and agent_name in llm_configs:
            model_name = llm_configs[agent_name]
        
        # Fallback to default
        if not model_name:
            model_name = 'qwen3-4b'
        
        step_latency = calculate_latency(input_tokens, output_tokens, model_name, latency_data)
        total_latency += step_latency
    
    return total_latency


# ============================================================================
# Pareto Frontier
# ============================================================================

def compute_pareto_frontier(
    points: List[Tuple[float, float]],
    maximize_x: bool = False,
    maximize_y: bool = True
) -> List[Tuple[float, float]]:
    """
    Compute Pareto frontier from a set of 2D points.
    
    Args:
        points: List of (x, y) tuples
        maximize_x: If True, maximize x; otherwise minimize
        maximize_y: If True, maximize y; otherwise minimize
    
    Returns:
        List of Pareto-optimal points sorted by x
    """
    if not points:
        return []
    
    # Convert to numpy for easier manipulation
    points_array = np.array(points)
    
    # Flip signs for minimization
    if not maximize_x:
        points_array[:, 0] = -points_array[:, 0]
    if not maximize_y:
        points_array[:, 1] = -points_array[:, 1]
    
    # Sort by x (descending for maximization)
    sorted_indices = np.argsort(-points_array[:, 0])
    sorted_points = points_array[sorted_indices]
    
    # Find Pareto frontier
    pareto_indices = []
    max_y = -np.inf
    
    for i, (x, y) in enumerate(sorted_points):
        if y > max_y:
            pareto_indices.append(sorted_indices[i])
            max_y = y
    
    # Get original points
    pareto_points = [points[i] for i in pareto_indices]
    
    # Sort by x for output
    pareto_points.sort(key=lambda p: p[0])
    
    return pareto_points


# ============================================================================
# Dictionary-based Pareto Frontier (for workflow results)
# ============================================================================

def compute_pareto_frontier_dict(
    points: List[Dict],
    accuracy_key: str = 'accuracy',
    latency_key: str = 'latency'
) -> List[Dict]:
    """
    Compute Pareto frontier for accuracy-latency trade-off from dictionaries.
    
    Maximizes accuracy, minimizes latency.
    Uses efficient O(n log n) algorithm.
    
    Args:
        points: List of dictionaries with accuracy and latency keys
        accuracy_key: Key for accuracy values
        latency_key: Key for latency values
    
    Returns:
        List of Pareto optimal points (dictionaries)
    """
    if not points:
        return []
    
    # Extract costs: [latency, -accuracy] so we minimize both
    costs = np.array([[p[latency_key], -p[accuracy_key]] for p in points])
    n_points = costs.shape[0]
    
    # Sort by latency
    sorted_idx = np.argsort(costs[:, 0])
    sorted_costs = costs[sorted_idx]
    
    is_efficient = np.zeros(n_points, dtype=bool)
    best_y = np.inf
    
    # Scan through - if second objective improves, it's Pareto optimal
    for i in range(n_points):
        if sorted_costs[i, 1] < best_y:
            is_efficient[sorted_idx[i]] = True
            best_y = sorted_costs[i, 1]
    
    return [points[i] for i in range(n_points) if is_efficient[i]]


# ============================================================================
# Statistical Utilities
# ============================================================================

def calculate_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate mean and confidence interval.
    
    Args:
        data: List of data points
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    import scipy.stats as stats
    
    data_array = np.array(data)
    mean = np.mean(data_array)
    sem = stats.sem(data_array)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data_array) - 1)
    
    return mean, mean - interval, mean + interval


# ============================================================================
# Optimization helpers
# ============================================================================

def is_pareto_efficient(costs):
    """
    Find Pareto efficient points.

    For 2D problems this uses an O(n log n) scan; otherwise a vectorized
    O(n^2) fallback is used.
    """
    n_points = costs.shape[0]

    if costs.shape[1] == 2:
        sorted_idx = np.argsort(costs[:, 0])
        sorted_costs = costs[sorted_idx]

        is_efficient = np.zeros(n_points, dtype=bool)
        best_y = np.inf
        for i in range(n_points):
            if sorted_costs[i, 1] < best_y:
                is_efficient[sorted_idx[i]] = True
                best_y = sorted_costs[i, 1]
        return is_efficient

    is_efficient = np.ones(n_points, dtype=bool)
    for i, c in enumerate(tqdm(costs, desc="Finding Pareto frontier")):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(
                costs[is_efficient] == c, axis=1
            )
    return is_efficient


def filter_pareto_optimal(df_subagent, accuracy_col="accuracy", latency_col="latency"):
    """Filter DataFrame to only Pareto-optimal rows."""
    if len(df_subagent) == 0:
        return df_subagent

    costs = np.column_stack([
        df_subagent[latency_col].values,
        -df_subagent[accuracy_col].values,
    ])
    pareto_mask = is_pareto_efficient(costs)
    return df_subagent[pareto_mask].reset_index(drop=True)


def get_model_marker(model_name, model_to_marker):
    """Get or assign plotting marker for a model name."""
    if model_name not in model_to_marker:
        markers = ["o", "s", "^", "D", "v", "p", "*", "h", "H", "<", ">", "X", "d", "P"]
        model_to_marker[model_name] = markers[len(model_to_marker) % len(markers)]
    return model_to_marker[model_name]


# ============================================================================
# Misc shared modeling helpers
# ============================================================================

def extract_model_size_key(model_name: str) -> float:
    """Extract numeric model size from names like `Qwen/Qwen3-8B` -> 8.0."""
    match = re.search(r"(\\d+(?:\\.\\d+)?)[Bb]", model_name)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract model size from model name: {model_name}")
