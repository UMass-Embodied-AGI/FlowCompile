"""
Result consolidation utilities for FlowCompile.

Handles loading and processing experiment results from config directories.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import os

from flowcompile.core.analysis.modeling import load_latency_data, extract_model_name
from flowcompile.core.llm.config import build_setting


# ============================================================================
# Config Processing
# ============================================================================

def _score_is_success(metric: str, score: float) -> bool:
    if metric == "f1":
        return score > 0.5
    return score >= 0.5

def calculate_latency_from_trace(
    trace_file: Path,
    latency_data: Dict,
    llm_configs: Dict,
    workflow_type: str = 'math',
    return_per_sample: bool = False
) -> Dict:
    """
    Calculate workflow latency across all samples in a trace file.
    Requires evaluated traces with unified score/metric fields.
    
    Args:
        trace_file: Path to trace.jsonl
        latency_data: Model latency data from load_latency_data()
        llm_configs: Mapping of agent names to model settings
        workflow_type: 'math', 'hotpotqa', or 'livecodebench' (affects which score field to use)
        return_per_sample: If True, include per-sample latency and score arrays
    
    Returns:
        Dictionary with:
        - 'mean_latency': Mean latency in seconds
        - 'per_sample_latency': List of per-sample latencies (if return_per_sample=True)
        - 'per_sample_score': List of per-sample scores (if return_per_sample=True)
        - 'problem_ids': List of problem IDs (if return_per_sample=True)
    """
    sample_latencies = []
    sample_scores = []
    problem_ids = []
    
    try:
        with open(trace_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                sample_latency = 0.0

                # Ensure evaluated traces include unified score/metric
                if "score" not in data or "metric" not in data:
                    raise ValueError(
                        f"Trace entry missing score/metric in {trace_file}. "
                        "Ensure validation has annotated traces before consolidation."
                    )
                
                if 'steps' in data:
                    for step in data['steps']:
                        if 'metadata' not in step:
                            continue
                        
                        metadata = step['metadata']
                        input_tokens = metadata.get('input_tokens', 0)
                        output_tokens = metadata.get('output_tokens', 0)
                        
                        # Get model for this agent
                        agent_name = step.get('agent', '')
                        if agent_name == "test":
                            continue

                        # Skip non-LLM/tool/noop steps (e.g., extract_answer).
                        step_type = metadata.get("type")
                        if step_type in {"rule_based", "noop"}:
                            continue

                        model_name = None
                        
                        if agent_name in llm_configs:
                            setting = llm_configs[agent_name]
                            model_name = extract_model_name(setting, return_hf_name=True)
                        else:
                            # If a step has no LLM mapping and no token usage, treat it as non-LLM.
                            if (not input_tokens) and (not output_tokens):
                                continue
                            raise ValueError(
                                f"Missing LLM config for tokenized step agent '{agent_name}'. "
                                f"Available agents: {list(llm_configs.keys())}"
                            )


                        if agent_name == "simple_math_solver":
                            config_temp = os.path.join(os.path.dirname(trace_file), "config_info.json")
                            uniform_setting = json.load(open(config_temp, 'r')).get('original_config', {}).get('uniform_setting', '')
                            if uniform_setting == "qwq-32b":
                                model_name = "Qwen/QwQ-32B"
                            elif uniform_setting == "ds-32b":
                                model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
                            else:
                                raise ValueError(f"Unknown uniform_setting '{uniform_setting}' for simple_math_solver in {trace_file}")
                        
                        if not model_name or model_name not in latency_data:
                            raise ValueError(f"Model name '{model_name}' for agent '{agent_name}' (original: {step.get('agent', '')}) not found in latency data. Available agents: {list(llm_configs.keys())}")
                    
                        
                        latency_info = latency_data[model_name]
                        prefill = input_tokens * latency_info['prefill_latency_per_token']
                        decode = output_tokens * latency_info['decode_latency_per_token']
                        sample_latency += prefill + decode
                
                sample_latencies.append(sample_latency)
                
                if return_per_sample:
                    metric = data.get("metric")
                    if metric is None or "score" not in data:
                        raise ValueError(
                            f"Trace entry missing score/metric in {trace_file}. "
                            "Ensure validation has annotated traces before consolidation."
                        )
                    sample_scores.append(float(data["score"]))
                    
                    # Get problem ID - use raw ID from original data or fallback to question
                    problem_id = None
                    
                    # First try top-level fields
                    problem_id = data.get('problem_id') or data.get('question_id') or data.get('id')
                    
                    # Try metadata
                    if problem_id is None:
                        problem_id = data.get('metadata', {}).get('problem_id')
                    
                    # Try original_sample
                    if problem_id is None:
                        orig = data.get('metadata', {}).get('original_sample', {})
                        problem_id = orig.get('problem_id') or orig.get('question_id') or orig.get('id') or orig.get('_id')
                    
                    # Fallback to question if no ID found
                    if problem_id is None:
                        problem_id = data.get('question') or data.get('metadata', {}).get('original_sample', {}).get('question')
                    
                    # Final fallback to index
                    if problem_id is None:
                        problem_id = f"sample_{len(problem_ids)}"
                    
                    problem_ids.append(str(problem_id))
    except FileNotFoundError as e:
        print(f"Warning: Trace file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in trace file: {e}")
        raise
    except Exception as e:
        print(f"Warning: Error processing trace file {trace_file}: {e}")
        import traceback
        traceback.print_exc()
        raise
    except:
        raise
    result = {
        'mean_latency': float(np.mean(sample_latencies)) if sample_latencies else 0.0
    }
    
    if return_per_sample:
        result['per_sample_latency'] = sample_latencies
        result['per_sample_score'] = sample_scores
        result['problem_ids'] = problem_ids
    
    return result


def _agents_to_settings_map(agents: Dict) -> Dict[str, str]:
    settings: Dict[str, str] = {}
    for agent_name, agent_cfg in (agents or {}).items():
        if isinstance(agent_cfg, dict):
            setting = agent_cfg.get("setting")
            if not setting:
                setting = build_setting(agent_cfg.get("model"), agent_cfg.get("budget"))
        else:
            setting = str(agent_cfg) if agent_cfg is not None else None
        if setting:
            settings[str(agent_name)] = str(setting)
    return settings


def _synthesize_legacy_setting_fields(agents: Dict) -> Dict[str, str]:
    """Backfill legacy *_setting fields from compiled v2 agents payload."""
    setting_by_agent = _agents_to_settings_map(agents)
    field_map = {
        "programmer": "programmer_setting",
        "refine_solver": "refine_setting",
        "detailed_solver": "detailed_setting",
        "generate_solver": "generate_setting",
        "sc_ensemble": "sc_ensemble_setting",
        "answer_generate": "answer_generate_setting",
        "format_answer": "format_answer_setting",
        "code_generate": "code_generate_setting",
        "reflection_test": "reflection_test_setting",
        "test": "test_setting",
    }
    synthesized = {
        field_name: setting_by_agent[agent_name]
        for agent_name, field_name in field_map.items()
        if setting_by_agent.get(agent_name)
    }
    # Older analysis code sometimes expects generate1_setting.
    if synthesized.get("generate_setting") and not synthesized.get("generate1_setting"):
        synthesized["generate1_setting"] = synthesized["generate_setting"]
    return synthesized


def _extract_predicted_metrics(
    original_config: Dict,
    config_results: Dict,
) -> Tuple[Optional[float], Optional[float]]:
    original_metrics = original_config.get("metrics") or {}
    result_metrics = config_results.get("metrics") or {}

    predicted_accuracy = original_config.get("workflow_accuracy")
    if predicted_accuracy is None:
        predicted_accuracy = original_metrics.get("expected_accuracy")
    if predicted_accuracy is None:
        predicted_accuracy = config_results.get("workflow_accuracy")
    if predicted_accuracy is None:
        predicted_accuracy = result_metrics.get("expected_accuracy")

    predicted_latency = original_config.get("workflow_latency")
    if predicted_latency is None:
        predicted_latency = original_metrics.get("expected_latency")
    if predicted_latency is None:
        predicted_latency = config_results.get("workflow_latency")
    if predicted_latency is None:
        predicted_latency = result_metrics.get("expected_latency")

    return predicted_accuracy, predicted_latency


def process_config_directory(
    config_dir: Path,
    latency_data: Dict,
    workflow_type: str = 'math',
    return_per_sample: bool = False,
    require_predicted_latency: bool = False
) -> Optional[Dict]:
    """
    Process a single config directory and extract all relevant information.
    
    Args:
        config_dir: Path to config_N directory
        latency_data: Model latency data
        workflow_type: 'math', 'hotpotqa', or 'livecodebench'
        return_per_sample: If True, include per-sample latency and score arrays
        require_predicted_latency: If True, error if predicted latency is missing
    
    Returns:
        Dictionary with config results, or None if invalid
    """
    config_info_file = config_dir / "config_info.json"
    config_results_file = config_dir / "config_results.json"
    trace_file = config_dir / "trace.jsonl"
    
    if not config_info_file.exists():
        return None
    
    with open(config_info_file, 'r') as f:
        config_info = json.load(f)
    
    # Also read config_results.json if it exists (contains actual_accuracy)
    config_results = {}
    if config_results_file.exists():
        with open(config_results_file, 'r') as f:
            config_results = json.load(f)
    
    original_config = config_info.get('original_config', {})
    llm_configs = config_info.get('llm_configs', {})
    if not llm_configs:
        llm_configs = _agents_to_settings_map(original_config.get("agents", {}))
    results = config_info.get('results', {})
    
    # Skip incomplete configs
    if not results:
        return None
    
    # Extract dynamic settings/counts from compiled config payload.
    settings = {
        key: value
        for key, value in original_config.items()
        if key.endswith('_setting') or key.endswith('_count')
    }
    agent_settings_map = _agents_to_settings_map(original_config.get("agents", {}))
    synthesized_settings = _synthesize_legacy_setting_fields(original_config.get("agents", {}))
    for key, value in synthesized_settings.items():
        settings.setdefault(key, value)
    uniform_setting = original_config.get('uniform_setting')
    if not uniform_setting and agent_settings_map:
        unique_agent_settings = sorted(set(agent_settings_map.values()))
        if len(unique_agent_settings) == 1:
            uniform_setting = unique_agent_settings[0]
    settings.update(
        {
            'uniform_setting': uniform_setting or 'N/A',
            'level': (original_config.get('level') or
                      original_config.get('_eval_level_filter') or
                      config_info.get('level_filter', 'N/A')),
            'split': config_info.get('split', 'N/A'),
            'structure_id': original_config.get('structure_id', 'N/A'),
            'total_branches': original_config.get('total_branches', 'N/A'),
            'is_full': original_config.get('is_full', 'N/A'),
        }
    )
    
    # Determine metric type
    is_hotpotqa = settings.get('answer_generate_setting') not in ['N/A', None]
    metric_type = 'f1' if is_hotpotqa else 'accuracy'
    
    # Calculate latency (and optionally per-sample data)
    actual_latency = None
    per_sample_data = {}
    if trace_file.exists():
        trace_result = calculate_latency_from_trace(
            trace_file, latency_data, llm_configs, workflow_type, return_per_sample
        )
        actual_latency = trace_result['mean_latency']
        if return_per_sample:
            per_sample_data = {
                'per_sample_latency': trace_result.get('per_sample_latency', []),
                'per_sample_score': trace_result.get('per_sample_score', []),
                'problem_ids': trace_result.get('problem_ids', [])
            }
    
    # Get actual metrics from config_results.json if available, otherwise from config_info results
    actual_accuracy = (
        config_results.get('actual_accuracy')
        or config_results.get('actual_score')
        or results.get('actual_accuracy')
        or results.get('actual_score')
    )
    actual_f1 = config_results.get('actual_f1') or results.get('actual_f1')

    predicted_workflow_accuracy, predicted_workflow_latency = _extract_predicted_metrics(
        original_config, config_results
    )

    # Check for predicted latency if required
    if require_predicted_latency and predicted_workflow_latency is None:
        raise ValueError(
            f"Missing predicted latency in config {config_dir}. "
            f"Required for routing strategy based on predicted latency."
        )
    
    result = {
        'config_index': config_info.get('config_index', -1),
        'config_dir': config_dir.name,
        'predicted_workflow_accuracy': predicted_workflow_accuracy,
        'predicted_workflow_latency': predicted_workflow_latency,
        'actual_accuracy': actual_accuracy,
        'actual_f1': actual_f1,
        'metric_type': metric_type,
        'actual_latency': actual_latency,
        'total_problems': results.get('total_problems'),
        'id': None,
    }
    result.update(settings)
    
    if return_per_sample:
        result.update(per_sample_data)
    
    return result


def consolidate_results(
    results_dirs: List[str],
    latency_file: str,
    dir_ids: Optional[Dict[str, str]] = None,
    exclude_folders: Optional[List[str]] = None,
    workflow_type: str = 'math',
    return_per_sample: bool = False,
    require_predicted_latency: bool = False
) -> List[Dict]:
    """
    Consolidate results from multiple experiment directories.
    
    Args:
        results_dirs: List of result directory paths
        latency_file: Path to latency benchmark file
        dir_ids: Optional mapping of directory paths to custom IDs
        exclude_folders: Optional list of folder names to exclude
        workflow_type: 'math', 'hotpotqa', or 'livecodebench'
        return_per_sample: If True, include per-sample latency and score arrays
        require_predicted_latency: If True, error if predicted latency is missing
    
    Returns:
        List of consolidated result dictionaries
    """
    latency_data = load_latency_data(latency_file)
    all_results = []
    dir_ids = dir_ids or {}
    exclude_folders = exclude_folders or []
    
    for results_dir in results_dirs:
        results_dir = Path(results_dir)
        
        if results_dir.name in exclude_folders:
            continue
        
        if not results_dir.exists():
            continue
        
        dir_id = dir_ids.get(str(results_dir), results_dir.name)
        config_dirs = sorted(results_dir.glob("config_*"))
        
        for config_dir in config_dirs:
            result = process_config_directory(
                config_dir, latency_data, workflow_type, return_per_sample, require_predicted_latency
            )
            if result:
                result['experiment'] = results_dir.name
                result['id'] = dir_id
                all_results.append(result)
    
    return all_results


# ============================================================================
# Result Discovery
# ============================================================================

def discover_experiment_dirs(
    base_dir: Path,
    folder_name: str,
    id_prefix: str = ""
) -> Tuple[List[str], Dict[str, str]]:
    """
    Auto-discover experiment subdirectories containing config_* folders.
    
    Args:
        base_dir: Base experiment directory
        folder_name: Name of folder to search (e.g., 'profile_workflow')
        id_prefix: Prefix for generated IDs
    
    Returns:
        Tuple of (list of subdirectory paths, dict of path->ID mappings)
    """
    folder = base_dir / folder_name
    subdirs = []
    dir_ids = {}
    
    if not folder.exists():
        return subdirs, dir_ids
    
    # Check for flat structure
    direct_configs = list(folder.glob("config_*"))
    if direct_configs and all(c.is_dir() for c in direct_configs):
        subdirs.append(str(folder))
        dir_ids[str(folder)] = f'{id_prefix}_test_all' if id_prefix else 'test_all'
        return subdirs, dir_ids
    
    # Nested structure
    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir() or not list(subdir.glob("config_*")):
            continue
        
        subdirs.append(str(subdir))
        dir_name = subdir.name
        
        # Determine ID based on name
        is_structure = 'with_structure' in dir_name or '_structure' in dir_name
        
        if 'test_all' in dir_name:
            base_id = 'test_all'
        elif 'test_by_level' in dir_name:
            base_id = 'test_by_level'
        elif 'validate_all' in dir_name:
            base_id = 'validate_all'
        elif 'validate_by_level' in dir_name:
            base_id = 'validate_by_level'
        else:
            base_id = dir_name
        
        if is_structure:
            base_id = f'{base_id}_structure'
        
        if id_prefix:
            dir_ids[str(subdir)] = f'{id_prefix}_{base_id}'
        else:
            dir_ids[str(subdir)] = base_id
    
    return subdirs, dir_ids


# ============================================================================
# Trace Loading for Level Routing
# ============================================================================

def load_uniform_traces(
    base_dir: Path,
    workflow_type: str = 'math'
) -> Dict[str, List[Dict]]:
    """
    Load trace.jsonl files from all config_* folders for level routing.
    
    Args:
        base_dir: Directory containing config_* subfolders
        workflow_type: 'math', 'hotpotqa', or 'livecodebench'
    
    Returns:
        Dict mapping model_name -> list of problem dicts with 'is_correct' (derived from score/metric), 'level', 'trace_data'
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return {}
    
    model_traces = {}
    config_dirs = sorted(base_dir.glob("config_*"))
    
    for config_dir in config_dirs:
        config_info_file = config_dir / "config_info.json"
        trace_file = config_dir / "trace.jsonl"
        
        if not config_info_file.exists() or not trace_file.exists():
            continue
        
        try:
            with open(config_info_file, 'r') as f:
                config_info = json.load(f)
            
            original_config = config_info.get('original_config', {})

            if workflow_type == 'livecodebench':
                setting = original_config.get('code_generate_setting', '')
            else:
                setting = original_config.get('uniform_setting', '')
            if not setting:
                synthesized = _synthesize_legacy_setting_fields(original_config.get("agents", {}))
                if workflow_type == 'livecodebench':
                    setting = synthesized.get("code_generate_setting", "")
                else:
                    # Prefer shared uniform setting if all agent settings are identical.
                    unique_settings = sorted(set(_agents_to_settings_map(original_config.get("agents", {})).values()))
                    setting = unique_settings[0] if len(unique_settings) == 1 else (unique_settings[0] if unique_settings else "")
            
            model_name = setting.replace('_budget_unlimited', '').replace('_budget_', '_') if setting else config_dir.name.replace('config_', '')
        except Exception:
            model_name = config_dir.name.replace('config_', '')
        
        problems = []
        with open(trace_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    metric = data.get("metric")
                    score = data.get("score")
                    if metric is None or score is None:
                        raise ValueError(
                            f"Trace entry missing score/metric in {trace_file}. "
                            "Ensure validation has annotated traces before consolidation."
                        )

                    is_correct = _score_is_success(metric, float(score))
                    if workflow_type == 'livecodebench':
                        level = data.get('metadata', {}).get('original_sample', {}).get('difficulty')
                    else:
                        level = data.get('level')
                        if level is None and 'metadata' in data:
                            level = data.get('metadata', {}).get('original_sample', {}).get('level')
                    
                    if level is not None:
                        problems.append({
                            'level': str(level),
                            'is_correct': is_correct,
                            'trace_data': data
                        })
                except json.JSONDecodeError:
                    continue
        
        if problems:
            model_traces[model_name] = problems
    
    return model_traces


# ==== merged from plotting.py ====

"""
Unified plotting utilities for FlowCompile analysis.

Provides a single PlotConfig-based interface for generating various comparison plots.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ============================================================================
# Color and Style Definitions
# ============================================================================

COLORS = {
    'workflow_all': '#2E86AB',
    'workflow_all_structure': '#85C1E9',
    'workflow_synthetic': '#F18F01',
    'workflow_synthetic_structure': '#F5B041',
    'uniform': '#C73E1D',
    'single_baseline': '#8B4513',
    'single_io': '#9370DB',
    'single_model': '#FF6347',
    'knn_k10': '#2ECC71',
    'knn_k20': '#E74C3C',
}

MARKERS = {
    'workflow_all': '*',
    'workflow_all_structure': 'P',
    'workflow_synthetic': 'o',
    'workflow_synthetic_structure': 'D',
    'uniform': 's',
    'single_baseline': 'X',
    'single_io': 'v',
    'single_model': '^',
    'knn_k10': 'h',
    'knn_k20': 'H',
}

COLORBLIND_PALETTE = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442', 
    '#0072B2', '#D55E00', '#CC79A7', '#999999'
]


def get_color_palette(n_colors: int) -> List[str]:
    """Get a colorblind-friendly palette."""
    if HAS_SEABORN:
        return sns.color_palette("colorblind", n_colors)
    return [COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)] for i in range(n_colors)]


# ============================================================================
# Data Point Extraction
# ============================================================================

@dataclass
class PlotData:
    """Container for plot data points."""
    latencies: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    
    def add(self, latency: float, accuracy: float, label: str = ""):
        self.latencies.append(latency)
        self.accuracies.append(accuracy)
        self.labels.append(label)
    
    def __len__(self):
        return len(self.latencies)
    
    def is_empty(self):
        return len(self.latencies) == 0


def extract_metric(result: Dict, workflow_type: str) -> Optional[float]:
    """Extract the appropriate metric (accuracy or F1) from a result."""
    metric_type = result.get('metric_type', 'accuracy')
    if metric_type == 'f1' or workflow_type == 'hotpotqa':
        return result.get('actual_f1')
    return result.get('actual_accuracy')


def has_structure_info(result: Dict) -> bool:
    """Check if result has structure information."""
    exp_name = result.get('experiment', result.get('experiment_name', ''))
    point_id = result.get('id', result.get('result_id', ''))
    return ('structure' in point_id or 
            'with_structure' in exp_name or 
            '_structure' in exp_name or 
            result.get('structure_id') not in [None, 'N/A'])


def is_all_sample_result(result: Dict) -> bool:
    """Treat missing/unknown level metadata as all-sample data."""
    level = result.get('level')
    if level in [None, 'N/A', 'unknown', 'all']:
        return True
    return str(level) == 'all'


def extract_model_from_setting(setting: str) -> tuple:
    """Extract model name and budget from setting string."""
    if not setting or setting == 'N/A':
        return 'unknown', 0
    base_name = setting.split('_budget_')[0] if '_budget_' in setting else setting
    budget = 0
    if '_budget_' in setting:
        try:
            budget_str = setting.split('_budget_')[1].split('_')[0].split('-')[0]
            budget = -1 if budget_str == 'unlimited' else int(budget_str)
        except (IndexError, ValueError):
            pass
    return base_name, budget


# ============================================================================
# Unified Plot Function
# ============================================================================

@dataclass
class PlotConfig:
    """Configuration for what to include in a comparison plot."""
    # Workflow options
    show_workflow_all: bool = True
    show_workflow_all_structure: bool = True
    show_workflow_synthetic: bool = True
    show_workflow_synthetic_structure: bool = True
    
    # Uniform config options
    show_uniform: bool = True
    max_budget_only: bool = False
    
    # Baseline options
    show_single_baseline: bool = False
    show_single_io: bool = False
    show_single_model: bool = False
    
    # Router options
    show_knn_k10: bool = False
    show_knn_k20: bool = False
    
    # Plot appearance
    title_suffix: str = ""
    figsize: tuple = (14, 10)


def plot_comparison(
    output_path: Path,
    workflow_type: str,
    exp_type: str,
    config: PlotConfig,
    workflow_results: Optional[List[Dict]] = None,
    workflow_all_results: Optional[List[Dict]] = None,
    synthetic_points: Optional[List[Dict]] = None,
    uniform_results: Optional[List[Dict]] = None,
    single_baseline_results: Optional[List[Dict]] = None,
    single_nothinking_results: Optional[List[Dict]] = None,
    single_model_results: Optional[List[Dict]] = None,
    knn_points: Optional[List[Dict]] = None,
) -> Path:
    """
    Generate a comparison plot based on the provided configuration.
    
    Args:
        output_path: Path to save the plot
        workflow_type: 'math', 'hotpotqa', or 'livecodebench'
        exp_type: 'test' or 'validate'
        config: PlotConfig specifying what to include
        workflow_results: Workflow evaluation results (with structure exploration)
        workflow_all_results: Workflow test_all results (same config for whole benchmark)
        synthetic_points: Pre-computed synthetic Pareto points
        uniform_results: Uniform config evaluation results
        single_baseline_results: Single-agent baseline results
        single_nothinking_results: Single-agent I/O baseline results
        single_model_results: Single model (no workflow) baseline results
        knn_points: KNN router results aggregated by (k, threshold)
    
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=config.figsize)
    legend_handles = []
    
    # 1. Workflow all-sample points
    if config.show_workflow_all or config.show_workflow_all_structure:
        all_points = PlotData()
        structure_points = PlotData()
        
        # Process workflow_all_results (test_all, no structure exploration)
        if config.show_workflow_all:
            for r in (workflow_all_results or []):
                if exp_type not in r.get('experiment', ''):
                    continue
                if not is_all_sample_result(r):
                    continue
                
                acc = extract_metric(r, workflow_type)
                lat = r.get('actual_latency')
                if acc is None or lat is None:
                    continue
                
                all_points.add(lat, acc)
        
        # Process workflow_results (with structure exploration)
        if config.show_workflow_all_structure:
            for r in (workflow_results or []):
                if exp_type not in r.get('experiment', ''):
                    continue
                if not is_all_sample_result(r):
                    continue
                
                acc = extract_metric(r, workflow_type)
                lat = r.get('actual_latency')
                if acc is None or lat is None:
                    continue
                
                if has_structure_info(r):
                    structure_points.add(lat, acc)
        
        if config.show_workflow_all and not all_points.is_empty():
            h = plt.scatter(all_points.latencies, all_points.accuracies,
                           c=COLORS['workflow_all'], marker=MARKERS['workflow_all'],
                           s=300, alpha=0.8, edgecolors='black', linewidth=2,
                           label=f'Model+Budget (Ours)', zorder=5)
            legend_handles.append(h)
        
        if config.show_workflow_all_structure and not structure_points.is_empty():
            h = plt.scatter(structure_points.latencies, structure_points.accuracies,
                           c=COLORS['workflow_all_structure'], marker=MARKERS['workflow_all_structure'],
                           s=300, alpha=0.8, edgecolors='black', linewidth=2,
                           label=f'Model+Budget+Structure (Ours)', zorder=5)
            legend_handles.append(h)
    
    # 2. Workflow synthetic points
    if config.show_workflow_synthetic or config.show_workflow_synthetic_structure:
        synthetic_no_struct = PlotData()
        synthetic_struct = PlotData()
        
        for p in (synthetic_points or []):
            acc = p.get('accuracy')
            lat = p.get('latency')
            if acc is None or lat is None:
                continue
            
            if has_structure_info(p):
                synthetic_struct.add(lat, acc)
            else:
                synthetic_no_struct.add(lat, acc)
        
        if config.show_workflow_synthetic and not synthetic_no_struct.is_empty():
            h = plt.scatter(synthetic_no_struct.latencies, synthetic_no_struct.accuracies,
                           c=COLORS['workflow_synthetic'], marker=MARKERS['workflow_synthetic'],
                           s=150, alpha=0.8, edgecolors='black', linewidth=1.5,
                           label=f'Workflow Synthetic Pareto (n={len(synthetic_no_struct)})', zorder=4)
            legend_handles.append(h)
        
        if config.show_workflow_synthetic_structure and not synthetic_struct.is_empty():
            h = plt.scatter(synthetic_struct.latencies, synthetic_struct.accuracies,
                           c=COLORS['workflow_synthetic_structure'], marker=MARKERS['workflow_synthetic_structure'],
                           s=150, alpha=0.8, edgecolors='black', linewidth=1.5,
                           label=f'Model+Budget+Structure Synthetic (Ours)', zorder=4)
            legend_handles.append(h)
    
    # 3. Uniform config points
    if config.show_uniform and uniform_results:
        model_points: Dict[str, List[Dict]] = {}
        
        for r in uniform_results:
            if exp_type not in r.get('experiment', ''):
                continue
            if not is_all_sample_result(r):
                continue
            
            acc = extract_metric(r, workflow_type)
            lat = r.get('actual_latency')
            if acc is None or lat is None:
                continue
            
            uniform_setting = r.get('uniform_setting') or r.get('code_generate_setting') or r.get('programmer_setting') or ''
            model_name, budget = extract_model_from_setting(uniform_setting)
            
            if model_name not in model_points:
                model_points[model_name] = []
            model_points[model_name].append({'accuracy': acc, 'latency': lat, 'budget': budget})
        
        # Filter to max budget only if requested
        if config.max_budget_only:
            for model_name in model_points:
                points = model_points[model_name]
                max_budget = max(p['budget'] for p in points)
                model_points[model_name] = [p for p in points if p['budget'] == max_budget]
        
        # Collect all points from all models to plot with same color/marker
        all_lats = []
        all_accs = []
        for model_name, points in model_points.items():
            for p in points:
                all_lats.append(p['latency'])
                all_accs.append(p['accuracy'])
        
        # Plot all uniform baseline points with same color and marker
        if all_lats:
            h = plt.scatter(all_lats, all_accs, c=COLORS['uniform'], marker=MARKERS['uniform'],
                           s=200, alpha=0.4, edgecolors='black', linewidth=1.5,
                           label=f'Fixed Model (Baseline)', zorder=3)
            legend_handles.append(h)
    
    # 4. Single-agent baseline
    if config.show_single_baseline and single_baseline_results:
        points = PlotData()
        for r in single_baseline_results:
            if exp_type not in r.get('experiment', ''):
                continue
            if not is_all_sample_result(r):
                continue
            acc = extract_metric(r, workflow_type)
            lat = r.get('actual_latency')
            if acc is not None and lat is not None:
                points.add(lat, acc)
        
        if not points.is_empty():
            h = plt.scatter(points.latencies, points.accuracies,
                           c=COLORS['single_baseline'], marker=MARKERS['single_baseline'],
                           s=200, alpha=0.8, edgecolors='black', linewidth=2,
                           label=f'Single-Agent Baseline (n={len(points)})', zorder=6)
            legend_handles.append(h)
    
    # 5. Single-agent I/O baseline
    if config.show_single_io and single_nothinking_results:
        points = PlotData()
        for r in single_nothinking_results:
            if exp_type not in r.get('experiment', ''):
                continue
            if not is_all_sample_result(r):
                continue
            acc = extract_metric(r, workflow_type)
            lat = r.get('actual_latency')
            if acc is not None and lat is not None:
                points.add(lat, acc)
        
        if not points.is_empty():
            h = plt.scatter(points.latencies, points.accuracies,
                           c=COLORS['single_io'], marker=MARKERS['single_io'],
                           s=200, alpha=0.8, edgecolors='black', linewidth=2,
                           label=f'Single-Agent I/O (n={len(points)})', zorder=6)
            legend_handles.append(h)
    
    # 5b. Single model (no workflow) baseline
    if config.show_single_model and single_model_results:
        # Find the result with largest model size
        def extract_model_size(result):
            """Extract model size in billions from setting names."""
            for key in ['code_generate_setting', 'generate_setting', 'answer_generate_setting', 'uniform_setting']:
                setting = result.get(key)
                if setting and setting not in ['N/A', 'Unknown', None]:
                    # Extract size from patterns like "qwen3-32b", "qwen3-8b", "qwen3-1.7b"
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)[bB]', setting)
                    if match:
                        return float(match.group(1))
            return 0.0
        
        best_result = None
        best_size = -1
        for r in single_model_results:
            if not is_all_sample_result(r):
                continue
            acc = extract_metric(r, workflow_type)
            lat = r.get('actual_latency')
            if acc is not None and lat is not None:
                size = extract_model_size(r)
                if size > best_size:
                    best_size = size
                    best_result = r
        
        if best_result is not None:
            acc = extract_metric(best_result, workflow_type)
            lat = best_result.get('actual_latency')
            
            # Extract model name from settings (check for N/A and None)
            def get_valid_setting(result, *keys):
                for key in keys:
                    val = result.get(key)
                    if val and val not in ['N/A', 'Unknown']:
                        return val
                return None
            
            model_name = get_valid_setting(
                best_result,
                'code_generate_setting',
                'generate_setting',
                'answer_generate_setting',
                'uniform_setting'
            ) or 'Unknown'
            
            # Clean up the model name (remove _budget_unlimited suffix)
            if model_name != 'Unknown' and '_budget_' in model_name:
                model_name = model_name.split('_budget_')[0]
            
            h = plt.scatter([lat], [acc],
                           c=COLORS['single_model'], marker=MARKERS['single_model'],
                           s=200, alpha=0.4, edgecolors='black', linewidth=2,
                           label=f'Single Model ({model_name}) (Baseline)', zorder=6)
            legend_handles.append(h)
    
    # 6. KNN router points
    if knn_points:
        if config.show_knn_k10:
            k10_points = [p for p in knn_points if p.get('k') == 10]
            if k10_points:
                lats = [p['latency'] for p in k10_points]
                accs = [p['accuracy'] for p in k10_points]
                h = plt.scatter(lats, accs, c=COLORS['knn_k10'], marker=MARKERS['knn_k10'],
                               s=180, alpha=0.75, edgecolors='black', linewidth=1.5,
                               label=f'Model+Budget+Structure+KNN Router k=10 (Ours)', zorder=7)
                legend_handles.append(h)
        
        if config.show_knn_k20:
            k20_points = [p for p in knn_points if p.get('k') == 20]
            if k20_points:
                lats = [p['latency'] for p in k20_points]
                accs = [p['accuracy'] for p in k20_points]
                h = plt.scatter(lats, accs, c=COLORS['knn_k20'], marker=MARKERS['knn_k20'],
                               s=180, alpha=0.75, edgecolors='black', linewidth=1.5,
                               label=f'Model+Budget+Structure+KNN Router k=20 (Ours)', zorder=7)
                legend_handles.append(h)
    
    # Finalize plot
    metric_name = 'F1' if workflow_type == 'hotpotqa' else 'Accuracy'
    plt.xlabel('Latency (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=14, fontweight='bold')
    
    title = f'{metric_name}-Latency Trade-off ({exp_type.capitalize()})'
    if config.title_suffix:
        title += f' {config.title_suffix}'
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.legend(fontsize=11, loc='best', framealpha=0.95, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


# ============================================================================
# Prediction vs Actual Plots
# ============================================================================

def plot_prediction_vs_actual(
    data: List[Dict],
    output_path: Path,
    metric_type: str = 'accuracy',
    workflow_type: str = 'math',
    title_suffix: str = ""
) -> Optional[Path]:
    """
    Plot predicted vs actual metrics with correlation statistics.
    
    Args:
        data: List of result dictionaries
        output_path: Path to save the plot
        metric_type: 'accuracy' or 'latency'
        workflow_type: 'math' or 'hotpotqa'
        title_suffix: Optional suffix for title
    
    Returns:
        Path to saved plot, or None if not enough data
    """
    from scipy import stats
    
    # Determine columns
    if metric_type == 'accuracy':
        actual_col = 'actual_f1' if workflow_type == 'hotpotqa' else 'actual_accuracy'
        predicted_col = 'predicted_workflow_accuracy'
        metric_name = 'F1' if workflow_type == 'hotpotqa' else 'Accuracy'
        color = '#2E86AB'
    else:
        actual_col = 'actual_latency'
        predicted_col = 'predicted_workflow_latency'
        metric_name = 'Latency'
        color = '#C73E1D'
    
    # Extract valid points
    valid_data = [
        (d[predicted_col], d[actual_col])
        for d in data
        if d.get(predicted_col) is not None and d.get(actual_col) is not None
    ]
    
    if len(valid_data) < 2:
        return None
    
    predicted = np.array([p[0] for p in valid_data])
    actual = np.array([p[1] for p in valid_data])
    
    # Calculate correlations
    spearman_corr, spearman_p = stats.spearmanr(predicted, actual)
    kendall_tau, _ = stats.kendalltau(predicted, actual)
    mae = np.mean(np.abs(predicted - actual))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.scatter(predicted, actual, s=100, alpha=0.7, edgecolors='black', linewidth=1.5, c=color)
    
    # Perfect prediction line
    min_val, max_val = min(predicted.min(), actual.min()), max(predicted.max(), actual.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction', alpha=0.7)
    
    # Regression line
    z = np.polyfit(predicted, actual, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(predicted.min(), predicted.max(), 100)
    plt.plot(x_line, p_line(x_line), 'g-', linewidth=2.5, label='Linear Fit', alpha=0.7)
    
    # Labels
    unit = ' (s)' if metric_type == 'latency' else ''
    plt.xlabel(f'Predicted {metric_name}{unit}', fontsize=14, fontweight='bold')
    plt.ylabel(f'Actual {metric_name}{unit}', fontsize=14, fontweight='bold')
    
    mae_fmt = '.2f' if metric_type == 'latency' else '.4f'
    title = (f'Predicted vs Actual {metric_name}{title_suffix} (n={len(valid_data)})\n'
             f'Spearman ρ={spearman_corr:.4f} (p={spearman_p:.2e}) | '
             f'Kendall τ={kendall_tau:.4f} | MAE={mae:{mae_fmt}}')
    plt.title(title, fontsize=13, fontweight='bold')
    
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
