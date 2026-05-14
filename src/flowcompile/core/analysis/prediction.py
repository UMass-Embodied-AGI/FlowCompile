"""Search-space parsing and filtering for workflow compilation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, List, Optional, Set, Tuple

import pandas as pd

from flowcompile.core.llm.config import parse_config


SEARCH_AXES = {"model", "budget", "structure"}


def _normalize_budget(value: Any) -> str:
    if value is None:
        return "__none__"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("none", "null", "__none__", ""):
            return "__none__"
        if lowered.isdigit():
            return str(int(lowered))
        return lowered
    if isinstance(value, (int, float)):
        if int(value) == value:
            return str(int(value))
        return str(value)
    return str(value).strip().lower()


def setting_to_model_budget(setting: str) -> Tuple[str, str]:
    model, budget = parse_config(setting)
    return model, _normalize_budget(budget)


def parse_search_axes(raw_axes: Optional[Iterable[str]]) -> Set[str]:
    if not raw_axes:
        return set(SEARCH_AXES)
    parsed: Set[str] = set()
    for axis in raw_axes:
        if axis is None:
            continue
        parts = [p.strip().lower() for p in str(axis).split(",") if p.strip()]
        parsed.update(parts)
    unknown = parsed - SEARCH_AXES
    if unknown:
        unknown_txt = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown search axis(es): {unknown_txt}. Allowed: {sorted(SEARCH_AXES)}")
    if not parsed:
        return set(SEARCH_AXES)
    return parsed


def parse_agent_constraints(entries: Optional[Iterable[str]], kind: str) -> Dict[str, Set[str]]:
    """Parse constraints like `agent=v1,v2` into map(agent -> set(values))."""
    parsed: Dict[str, Set[str]] = {}
    if not entries:
        return parsed
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --search-agent-{kind} entry '{entry}'. Expected format agent=v1,v2"
            )
        agent, raw_values = entry.split("=", 1)
        agent_key = agent.strip()
        if not agent_key:
            raise ValueError(f"Invalid --search-agent-{kind} entry '{entry}': empty agent name")
        values = {v.strip() for v in raw_values.split(",") if v.strip()}
        if not values:
            raise ValueError(f"Invalid --search-agent-{kind} entry '{entry}': empty value list")
        if kind == "budgets":
            values = {_normalize_budget(v) for v in values}
        parsed[agent_key] = values
    return parsed


@dataclass
class SearchSpaceSpec:
    search_axes: Set[str] = field(default_factory=lambda: set(SEARCH_AXES))
    models: Optional[Set[str]] = None
    budgets: Optional[Set[str]] = None
    structures: Optional[Set[str]] = None
    agent_models: Dict[str, Set[str]] = field(default_factory=dict)
    agent_budgets: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "SearchSpaceSpec":
        if not payload:
            return cls()
        search_axes = parse_search_axes(payload.get("search_axes"))
        models = payload.get("models")
        budgets = payload.get("budgets")
        structures = payload.get("structures")
        agent_models = payload.get("agent_models") or {}
        agent_budgets = payload.get("agent_budgets") or {}

        model_set = {m.strip() for m in models if str(m).strip()} if models else None
        budget_set = {_normalize_budget(b) for b in budgets} if budgets else None
        structure_set = {s.strip() for s in structures if str(s).strip()} if structures else None

        parsed_agent_models = {
            agent: {m.strip() for m in values if str(m).strip()}
            for agent, values in agent_models.items()
        }
        parsed_agent_budgets = {
            agent: {_normalize_budget(v) for v in values}
            for agent, values in agent_budgets.items()
        }
        return cls(
            search_axes=search_axes,
            models=model_set,
            budgets=budget_set,
            structures=structure_set,
            agent_models=parsed_agent_models,
            agent_budgets=parsed_agent_budgets,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_axes": sorted(self.search_axes),
            "models": sorted(self.models) if self.models is not None else None,
            "budgets": sorted(self.budgets) if self.budgets is not None else None,
            "structures": sorted(self.structures) if self.structures is not None else None,
            "agent_models": {k: sorted(v) for k, v in self.agent_models.items()},
            "agent_budgets": {k: sorted(v) for k, v in self.agent_budgets.items()},
        }


def apply_search_space_to_subagents(
    df_subagents: Dict[str, pd.DataFrame],
    required_agents: List[str],
    spec: SearchSpaceSpec,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    unknown_model_agents = set(spec.agent_models.keys()) - set(required_agents)
    unknown_budget_agents = set(spec.agent_budgets.keys()) - set(required_agents)
    if unknown_model_agents:
        raise ValueError(f"Unknown agent(s) in model constraints: {sorted(unknown_model_agents)}")
    if unknown_budget_agents:
        raise ValueError(f"Unknown agent(s) in budget constraints: {sorted(unknown_budget_agents)}")

    filtered: Dict[str, pd.DataFrame] = {}
    model_cardinality: Dict[str, int] = {}
    budget_cardinality: Dict[str, int] = {}
    resolved_locks: Dict[str, Dict[str, str]] = {}

    for agent in required_agents:
        if agent not in df_subagents:
            raise ValueError(f"Missing subagent data for '{agent}'")
        source = df_subagents[agent].copy()
        if source.empty:
            raise ValueError(f"No profiling configurations found for subagent '{agent}'")

        models: List[str] = []
        budgets: List[str] = []
        for setting in source["setting"].tolist():
            model, budget = setting_to_model_budget(setting)
            models.append(model)
            budgets.append(budget)
        source["_model"] = models
        source["_budget"] = budgets

        allowed_models = spec.agent_models.get(agent, spec.models)
        if allowed_models is not None:
            source = source[source["_model"].isin(allowed_models)]
        allowed_budgets = spec.agent_budgets.get(agent, spec.budgets)
        if allowed_budgets is not None:
            source = source[source["_budget"].isin(allowed_budgets)]

        if source.empty:
            raise ValueError(
                f"Search-space filtering removed all configs for subagent '{agent}'. "
                "Adjust model/budget constraints."
            )

        unique_models = sorted(source["_model"].unique().tolist())
        unique_budgets = sorted(source["_budget"].unique().tolist())
        model_cardinality[agent] = len(unique_models)
        budget_cardinality[agent] = len(unique_budgets)

        if "model" not in spec.search_axes and len(unique_models) != 1:
            raise ValueError(
                f"Model axis disabled but subagent '{agent}' has {len(unique_models)} models after filtering: "
                f"{unique_models}. Provide explicit model lock."
            )
        if "budget" not in spec.search_axes and len(unique_budgets) != 1:
            raise ValueError(
                f"Budget axis disabled but subagent '{agent}' has {len(unique_budgets)} budgets after filtering: "
                f"{unique_budgets}. Provide explicit budget lock."
            )

        lock_info: Dict[str, str] = {}
        if "model" not in spec.search_axes:
            lock_info["model"] = unique_models[0]
        if "budget" not in spec.search_axes:
            lock_info["budget"] = unique_budgets[0]
        if lock_info:
            resolved_locks[agent] = lock_info

        filtered[agent] = source.drop(columns=["_model", "_budget"]).reset_index(drop=True)

    diagnostics = {
        "model_cardinality": model_cardinality,
        "budget_cardinality": budget_cardinality,
        "resolved_locks": resolved_locks,
    }
    return filtered, diagnostics


def apply_structure_constraints(
    structures: List[Dict[str, Any]],
    spec: SearchSpaceSpec,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    def _is_multi_branch(structure: Dict[str, Any]) -> bool:
        branches = structure.get("total_branches")
        if branches is None:
            return True
        try:
            return int(branches) >= 2
        except (TypeError, ValueError):
            return True

    filtered = structures
    if spec.structures is not None:
        filtered = [s for s in structures if s.get("structure_id") in spec.structures]
    else:
        # Default policy: ignore single-branch workflows unless caller explicitly
        # requests structure IDs via search_space.structures.
        filtered = [s for s in filtered if _is_multi_branch(s)]

    if not filtered:
        raise ValueError("Structure filtering removed all workflow structures.")

    if "structure" not in spec.search_axes and len(filtered) != 1:
        ids = [s.get("structure_id") for s in filtered]
        raise ValueError(
            f"Structure axis disabled but {len(filtered)} structures remain: {ids}. "
            "Provide explicit single structure selection."
        )

    info: Dict[str, Any] = {
        "structure_count": len(filtered),
        "exclude_single_branch_default": spec.structures is None,
    }
    if "structure" not in spec.search_axes:
        info["resolved_structure"] = filtered[0].get("structure_id")
    return filtered, info


__all__ = [
    "SearchSpaceSpec",
    "SEARCH_AXES",
    "parse_search_axes",
    "parse_agent_constraints",
    "setting_to_model_budget",
    "apply_search_space_to_subagents",
    "apply_structure_constraints",
]


# ==== merged from workflow_metrics.py ====

"""Unified workflow structure and metric helpers."""

from typing import Dict, Any, List

import pandas as pd

from flowcompile.workflows.dsl_registry import get_workflow_module


def _normalize_workflow_type(workflow_type: str) -> str:
    normalized = (workflow_type or "").lower()
    if normalized in ("math500", "math-500"):
        return "math"
    return normalized


def enumerate_workflow_structures(workflow_type: str) -> List[Dict[str, Any]]:
    normalized = _normalize_workflow_type(workflow_type)
    workflow_module = get_workflow_module(normalized)
    return list(workflow_module.enumerate_structures())


def calculate_workflow_metrics(payload: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise TypeError("Workflow metrics payload must be a dict.")
    workflow_type = payload.get("workflow_type")
    if workflow_type is None:
        raise ValueError("Workflow metrics payload must include 'workflow_type'.")
    if "structure" not in payload:
        raise ValueError("Workflow metrics payload must include 'structure'.")

    normalized = _normalize_workflow_type(workflow_type)
    workflow_module = get_workflow_module(normalized)
    backward_payload = {k: v for k, v in payload.items() if k != "workflow_type"}
    return workflow_module.backward(backward_payload)


def calculate_workflow_accuracy(payload: Dict[str, Any]):
    if not isinstance(payload, dict):
        raise TypeError("Workflow accuracy payload must be a dict.")
    workflow_type = payload.get("workflow_type")
    if workflow_type is None:
        raise ValueError("Workflow accuracy payload must include 'workflow_type'.")
    if "structure" not in payload:
        raise ValueError("Workflow accuracy payload must include 'structure'.")

    normalized = _normalize_workflow_type(workflow_type)
    workflow_module = get_workflow_module(normalized)
    workflow_structure = payload["structure"]

    parsed_probs: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in ("workflow_type", "structure"):
            continue
        agent = key[2:] if key.startswith("p_") else key
        if agent == "fix_code":
            agent = "reflection_test"
        parsed_probs[agent] = value

    defaults = {
        "sc_ensemble": 1.0,
        "format_answer": 1.0,
    }

    metrics_payload: Dict[str, pd.DataFrame] = {}
    for agent in workflow_module.infer_agent_names():
        prob = parsed_probs.get(agent, defaults.get(agent, 0.0))
        metrics_payload[agent] = pd.DataFrame(
            {"setting": ["__single__"], "accuracy": [prob], "latency": [0.0]}
        )

    df = workflow_module.backward({"structure": workflow_structure, "metrics": metrics_payload})
    if df is None or df.empty:
        return 0.0
    return float(df["workflow_accuracy"].iloc[0])


__all__ = [
    "enumerate_workflow_structures",
    "calculate_workflow_metrics",
    "calculate_workflow_accuracy",
]


# ==== merged from prediction.py ====

"""
Workflow prediction core logic.

This module provides the core functionality for predicting optimal workflow configurations
by analyzing sub-agent performance data and exploring different workflow structures.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from flowcompile.core.analysis.modeling import (
    extract_model_name,
    extract_model_size_key,
    filter_pareto_optimal,
    is_pareto_efficient,
)


# Benchmark configuration table - defines structure and agents for each workflow
BENCHMARK_CONFIGS = {
    'math': {
        'agents': ['programmer', 'refine', 'detailed', 'generate', 'sc_ensemble'],
        'agent_mapping': {
            'programmer': 'programmer',
            'refine': 'refine_solver',
            'detailed': 'detailed_solver',
            'generate': 'generate_solver',
            'sc_ensemble': 'sc_ensemble'
        }
    },
    'hotpotqa': {
        'agents': ['answer_generate', 'sc_ensemble', 'format_answer'],
        'agent_mapping': {
            'answer_generate': 'answer_generate',
            'sc_ensemble': 'sc_ensemble',
            'format_answer': 'format_answer'
        }
    },
    'livecodebench': {
        'agents': ['code_generate', 'sc_ensemble', 'reflection_test'],
        'agent_mapping': {
            'code_generate': 'code_generate',
            'sc_ensemble': 'sc_ensemble',
            'reflection_test': 'reflection_test'
        }
    },
}


def load_and_prepare_data(data_file, workflow_type, args):
    """
    Load consolidated data and prepare DataFrames by level.
    
    Args:
        data_file: Path to consolidated data JSON file
        workflow_type: Workflow type ('math', 'hotpotqa', or 'livecodebench')
        args: Command-line arguments namespace
    
    Returns:
        Tuple of (df, levels, level_sample_counts, df_all_agg, df_all_subagents_dict)
    """
    # Load data
    with open(data_file, 'r') as f:
        all_data = json.load(f)
    
    df = pd.DataFrame(all_data)
    df['subagent_name'] = df['subagent']
    
    # Verify that required agents are present using config table
    available_agents = set(df['subagent_name'].unique())
    
    if workflow_type not in BENCHMARK_CONFIGS:
        raise ValueError(f"Invalid workflow type: {workflow_type}")
    
    config = BENCHMARK_CONFIGS[workflow_type]
    required_agents = set(config['agent_mapping'].values())
    
    missing_agents = required_agents - available_agents
    if missing_agents:
        raise ValueError(f"Missing required agents for {workflow_type} workflow: {missing_agents}")
    
    # Check if level column exists and has meaningful data
    has_levels = 'level' in df.columns and df['level'].notna().any()
    
    if has_levels:
        levels = sorted(df['level'].unique())
    else:
        levels = ['all']
    
    level_sample_counts = {}
    
    # Skip level-based split if requested or if no level data exists
    no_split = getattr(args, 'no_split_by_level', False) or not has_levels
    if no_split:
        levels = ['all']
    
    # Determine unique identifier column for counting samples
    if 'problem' in df.columns:
        id_column = 'problem'  # MATH workflow
    elif '_id' in df.columns:
        id_column = '_id'  # HotpotQA workflow
    elif 'question_id' in df.columns:
        id_column = 'question_id'  # LiveCodeBench workflow
    else:
        id_column = None
    
    # Pre-compute all-levels aggregate data for use as fallback when by-level data is missing
    df_all_agg = df.groupby(['subagent_name', 'setting']).agg({
        'accuracy': 'mean',
        'latency': 'mean'
    }).reset_index()
    
    # Split all-levels data by sub-agent based on workflow type
    df_all_subagents = {}
    for agent_name in config['agents']:
        agent_key = config['agent_mapping'][agent_name]
        df_all_subagents[agent_name] = df_all_agg[df_all_agg['subagent_name'] == agent_key].reset_index(drop=True)
    
    return df, levels, level_sample_counts, df_all_agg, df_all_subagents, id_column, has_levels


def prepare_level_dataframes(df, level, workflow_type, config, args, 
                             df_all_subagents, id_column, level_sample_counts, has_levels):
    """
    Prepare DataFrames for a specific level.
    
    Args:
        df: Full DataFrame
        level: Level to process
        workflow_type: Workflow type
        config: Benchmark configuration
        args: Command-line arguments
        df_all_subagents: Dictionary of all-levels DataFrames for each subagent
        id_column: Column name for unique sample identification
        level_sample_counts: Dictionary to track sample counts
        has_levels: Whether level data exists
    
    Returns:
        Dictionary of DataFrames for each subagent, and metadata
    """
    # Filter by level
    if level == 'all' or not has_levels:
        df_level = df.copy()
    else:
        df_level = df[df['level'] == level].copy()
    
    # Count unique samples for this level
    if id_column:
        n_samples = df_level[id_column].nunique()
    else:
        n_samples = len(df_level)
    level_sample_counts[level] = n_samples
    
    # Aggregate metrics
    df_level_agg = df_level.groupby(['subagent_name', 'setting']).agg({
        'accuracy': 'mean',
        'latency': 'mean'
    }).reset_index()
    
    # Split data by sub-agent
    df_subagents = {}
    for agent_name in config['agents']:
        agent_key = config['agent_mapping'][agent_name]
        df_subagents[agent_name] = df_level_agg[df_level_agg['subagent_name'] == agent_key].reset_index(drop=True)
    
    # Exclude small models from sc_ensemble if requested (only for non-uniform config)
    exclude_small = getattr(args, 'exclude_small_models', False)
    uniform_config = getattr(args, 'uniform_config', False)
    
    if exclude_small and not uniform_config and 'sc_ensemble' in df_subagents:
        df_subagents['sc_ensemble'] = df_subagents['sc_ensemble'][
            ~df_subagents['sc_ensemble']['setting'].str.contains('qwen3-0.6b|qwen3-1.7b', case=False, na=False)
        ].reset_index(drop=True)
    
    # For uniform config mode, filter to only settings that exist across ALL sub-agents
    if uniform_config:
        common_settings = set.intersection(*[set(df['setting']) for df in df_subagents.values()])
        print(f"  Found {len(common_settings)} common settings across all sub-agents")
        
        # Filter each dataframe to only include common settings
        for agent_name in df_subagents:
            df_subagents[agent_name] = df_subagents[agent_name][
                df_subagents[agent_name]['setting'].isin(common_settings)
            ].reset_index(drop=True)
    
    # Print configuration counts
    print(f"\nAggregated configuration counts for level {level}:")
    for agent_name in config['agents']:
        agent_display = config['agent_mapping'][agent_name]
        print(f"  {agent_display}: {len(df_subagents[agent_name])}")
    
    # Filter to max budget per model if requested (for all-level-max-budget mode)
    # IMPORTANT: This must run BEFORE Pareto filtering to ensure we keep max-budget configs
    max_budget_only = getattr(args, 'max_budget_only', False)
    if max_budget_only and not uniform_config:
        print(f"  Filtering to max budget per model...")
        for agent_name in df_subagents:
            df_agent = df_subagents[agent_name].copy()
            if len(df_agent) == 0:
                continue
            
            # Extract model and budget from setting column (format: "model_budget_value")
            def extract_model_budget(setting):
                parts = setting.rsplit('_budget_', 1)
                if len(parts) == 2:
                    model = parts[0]
                    budget_str = parts[1]
                    # Parse budget value (higher is better)
                    try:
                        budget = float(budget_str)
                    except ValueError:
                        budget = 0.0  # Non-numeric budgets get lowest priority
                else:
                    model = setting
                    budget = 0.0
                return model, budget
            
            df_agent[['_model', '_budget']] = df_agent['setting'].apply(
                lambda s: pd.Series(extract_model_budget(s))
            )
            
            # For each model, keep only the configuration with the maximum budget
            idx_max_budget = df_agent.groupby('_model')['_budget'].idxmax()
            df_agent = df_agent.loc[idx_max_budget].reset_index(drop=True)
            
            # Clean up temporary columns
            df_agent = df_agent.drop(['_model', '_budget'], axis=1)
            df_subagents[agent_name] = df_agent
            
        # Print updated configuration counts
        print(f"  After max-budget filtering:")
        for agent_name in config['agents']:
            agent_display = config['agent_mapping'][agent_name]
            print(f"    {agent_display}: {len(df_subagents[agent_name])}")
    
    # Apply Pareto pre-filtering if requested (after max-budget filtering)
    disable_pareto = getattr(args, 'disable_pareto_only', False)
    if not disable_pareto and not uniform_config:
        for agent_name in df_subagents:
            df_subagents[agent_name] = filter_pareto_optimal(df_subagents[agent_name])
    
    # Handle missing sub-agent data for by-level analysis
    if level != 'all':
        for agent_name in df_subagents:
            if len(df_subagents[agent_name]) == 0:
                agent_display = config['agent_mapping'][agent_name]
                print(f"⚠ Warning: No {agent_display} data for level {level}. Using all-levels data as estimate.")
                df_subagents[agent_name] = df_all_subagents[agent_name].copy()

    # Apply model/budget search-space constraints to sub-agent options.
    search_spec = SearchSpaceSpec.from_dict(getattr(args, "search_space", None))
    required_agents = list(config["agents"])
    df_subagents, search_info = apply_search_space_to_subagents(
        df_subagents,
        required_agents=required_agents,
        spec=search_spec,
    )
    if uniform_config:
        common_settings = set.intersection(*[set(df_agent["setting"]) for df_agent in df_subagents.values()])
    
    metadata = {
        'n_samples': n_samples,
        'common_settings': common_settings if uniform_config else None,
        'search_space_resolved': search_info,
    }
    
    return df_subagents, metadata


def compute_full_workflow_metrics(workflow_type, config, df_subagents, args):
    """
    Compute full workflow metrics using unified calculation functions.
    
    Args:
        workflow_type: Workflow type
        config: Benchmark configuration
        df_subagents: Dictionary of DataFrames for each subagent
        args: Command-line arguments
    
    Returns:
        DataFrame with workflow configurations and metrics
    """
    from flowcompile.workflows.dsl_registry import get_workflow_module
    
    uniform_config = getattr(args, 'uniform_config', False)
    
    if uniform_config:
        # Uniform config: compute workflows for each common setting
        return compute_uniform_workflow_metrics(workflow_type, config, df_subagents, args)

    workflow_module = get_workflow_module(workflow_type)
    
    search_spec = SearchSpaceSpec.from_dict(getattr(args, "search_space", None))
    if search_spec.structures is not None or "structure" not in search_spec.search_axes:
        all_structures = workflow_module.enumerate_structures()
        selected_structures, _ = apply_structure_constraints(all_structures, search_spec)
    else:
        selected_structures = [workflow_module.get_full_structure()]

    metrics_payload = {}
    for agent in workflow_module.infer_agent_names():
        if agent not in df_subagents:
            raise ValueError(f"Missing subagent data for '{agent}'")
        metrics_payload[agent] = df_subagents[agent]

    workflow_dfs = [
        workflow_module.backward({"structure": structure, "metrics": metrics_payload})
        for structure in selected_structures
    ]
    workflow_dfs = [df for df in workflow_dfs if df is not None and len(df) > 0]
    if not workflow_dfs:
        return pd.DataFrame()
    return pd.concat(workflow_dfs, ignore_index=True)


def compute_uniform_workflow_metrics(workflow_type, config, df_subagents, args):
    """
    Compute workflow metrics for uniform configuration mode.
    
    In uniform mode, all sub-agents use the same configuration setting.
    """
    from flowcompile.workflows.dsl_registry import get_workflow_module
    
    # Get common settings from metadata
    common_settings = set.intersection(*[set(df['setting']) for df in df_subagents.values()])
    workflow_results = []
    workflow_module = get_workflow_module(workflow_type)
    structure = workflow_module.get_full_structure()

    for setting in common_settings:
        metrics_payload = {}
        valid = True
        for agent in workflow_module.infer_agent_names():
            if agent not in df_subagents:
                valid = False
                break
            row = df_subagents[agent][df_subagents[agent]['setting'] == setting]
            if row.empty:
                valid = False
                break
            metrics_payload[agent] = row[["setting", "accuracy", "latency"]].iloc[[0]].reset_index(drop=True)
        if not valid:
            continue
        metrics_df = workflow_module.backward({"structure": structure, "metrics": metrics_payload})
        if metrics_df is None or metrics_df.empty:
            continue
        result = metrics_df.iloc[0].to_dict()
        result["uniform_setting"] = setting
        workflow_results.append(result)
    
    return pd.DataFrame(workflow_results)


def explore_workflow_structures(workflow_type, config, df_subagents, metadata, args):
    """
    Explore different workflow structures (pruned configurations).
    
    Args:
        workflow_type: Workflow type
        config: Benchmark configuration
        df_subagents: Dictionary of DataFrames for each subagent
        metadata: Metadata dictionary
        args: Command-line arguments
    
    Returns:
        Tuple of (all_structures_list, pruned_structures_dataframes)
    """
    from flowcompile.workflows.dsl_registry import get_workflow_module
    del metadata
    
    workflow_module = get_workflow_module(workflow_type)
    full_structure = workflow_module.get_full_structure()
    full_structure_id = full_structure['structure_id']
    all_structures = workflow_module.enumerate_structures()
    search_spec = SearchSpaceSpec.from_dict(getattr(args, "search_space", None))
    all_structures, _ = apply_structure_constraints(all_structures, search_spec)
    
    # Filter to only pruned structures (exclude full workflow)
    workflow_structures = [s for s in all_structures if s['structure_id'] != full_structure_id]
    
    # Filter to only structures with total_branches >= 2 (multi-branch structures)
    original_count = len(workflow_structures)
    workflow_structures = [s for s in workflow_structures if s.get('total_branches', 0) >= 2]
    
    print(f"\n  Exploring {len(workflow_structures)} pruned workflow structures...")
    print(f"  (Excluded '{full_structure_id}' as it's already computed)")
    if len(workflow_structures) < original_count:
        print(f"  (Filtered out {original_count - len(workflow_structures)} single-branch structures, keeping only total_branches >= 2)")
    
    # Evaluate each structure
    structure_results = {}
    for structure in tqdm(workflow_structures, desc="Evaluating structures"):
        structure_id = structure['structure_id']
        
        # Calculate metrics for this structure
        metrics_payload = {}
        missing_required = False
        for agent in workflow_module.infer_agent_names():
            if agent not in df_subagents:
                missing_required = True
                break
            metrics_payload[agent] = df_subagents[agent]
        if missing_required:
            continue
        structure_df = workflow_module.backward({"structure": structure, "metrics": metrics_payload})
        if len(structure_df) > 0:
            structure_results[structure_id] = {
                'structure': structure,
                'configs': structure_df,
                'num_configs': len(structure_df),
                'total_branches': structure['total_branches'],
                'best_accuracy': structure_df['workflow_accuracy'].max(),
                'min_latency': structure_df['workflow_latency'].min()
            }
    
    return all_structures, structure_results


def create_combined_plot(all_level_results, output_dir, args):
    """
    Create combined plot showing Pareto frontiers for all levels.
    
    Args:
        all_level_results: Dictionary of results for each level
        output_dir: Output directory for plots
        args: Command-line arguments
    """
    plt.figure(figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    prune_workflow = getattr(args, 'prune_workflow', False)
    
    for i, level in enumerate(sorted(all_level_results.keys())):
        result = all_level_results[level]
        
        # Use combined Pareto when structure exploration is enabled
        if prune_workflow and 'combined_all_pareto' in result:
            df_pareto = result['combined_all_pareto']
        else:
            df_pareto = result['workflow_pareto']
        
        if len(df_pareto) > 0:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Sort by latency for line plot
            df_sorted = df_pareto.sort_values('workflow_latency')
            
            plt.scatter(df_sorted['workflow_latency'], df_sorted['workflow_accuracy'],
                       c=color, marker=marker, s=150, alpha=0.7,
                       label=f'Level {level}', zorder=5)
            plt.plot(df_sorted['workflow_latency'], df_sorted['workflow_accuracy'],
                    c=color, alpha=0.4, linewidth=2, linestyle='--', zorder=4)
    
    plt.xlabel('Workflow Latency (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Workflow Accuracy', fontsize=14, fontweight='bold')
    
    title = 'Workflow Accuracy vs Latency - Pareto Frontier by Level'
    if prune_workflow:
        title += ' (With Structure Exploration)'
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_prefix = getattr(args, 'output_prefix', 'workflow_by_level')
    output_file = os.path.join(output_dir, f'{output_prefix}_combined.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved to {output_file}")
    plt.close()


def save_prediction_results(all_level_results, output_dir, args):
    """
    Save prediction results to JSON files.
    
    Args:
        all_level_results: Dictionary of results for each level
        output_dir: Output directory
        args: Command-line arguments
    """
    output_prefix = getattr(args, 'output_prefix', 'workflow_by_level')
    save_all = getattr(args, 'save_all_configs', False)
    prune_workflow = getattr(args, 'prune_workflow', False)
    
    # Prepare results data
    results_data = {}
    for level in sorted(all_level_results.keys()):
        result = all_level_results[level]
        
        # Determine which configs to save based on mode
        if prune_workflow and 'combined_all_pareto' in result:
            # When structure exploration is enabled, save combined Pareto (all structures)
            configs_to_save = result['combined_all_pareto']
            n_total = len(result.get('combined_all_configs', configs_to_save))
            n_pareto = len(configs_to_save)
        elif save_all and 'workflow_configs' in result:
            configs_to_save = result['workflow_configs']
            n_total = len(configs_to_save)
            n_pareto = len(result['workflow_pareto'])
        else:
            configs_to_save = result['workflow_pareto']
            n_total = len(result.get('workflow_configs', configs_to_save))
            n_pareto = len(configs_to_save)
        
        level_data = {
            'n_samples': result.get('n_samples', 0),
            'n_pareto_configs': n_pareto,
            'n_total_configs': n_total,
            'best_accuracy': float(configs_to_save['workflow_accuracy'].max()) if len(configs_to_save) > 0 else 0,
            'min_latency': float(configs_to_save['workflow_latency'].min()) if len(configs_to_save) > 0 else 0,
            'configs': configs_to_save.replace({np.nan: None}).to_dict('records')
        }
        
        results_data[str(level)] = level_data
    
    # Save main results
    output_json = os.path.join(output_dir, f'{output_prefix}_results.json')
    with open(output_json, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Results saved to {output_json}")
    
    # Note: We no longer save per-structure analysis since structure is treated
    # as just another configuration parameter (like model choice)


# ==== merged from prediction_wrapper.py ====

"""
Simplified workflow prediction functions that delegate to flowcompile modules.

This module provides a clean interface to the workflow prediction pipeline,
keeping the complete pipeline module minimal and maintainable.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from flowcompile.core.analysis.modeling import is_pareto_efficient


def predict_workflow_by_level_simplified(data_file, output_dir, args, workflow_type):
    """
    Simplified workflow prediction function that delegates to flowcompile modules.
    
    This is a much cleaner version that uses the modular architecture.
    
    Args:
        data_file: Path to consolidated data JSON file
        output_dir: Directory for output files
        args: Command-line arguments namespace
        workflow_type: Workflow type ('math', 'hotpotqa', or 'livecodebench')
    """
    print(f"\n{'='*80}")
    print(f"WORKFLOW PREDICTION: {workflow_type.upper()}")
    print(f"{'='*80}\n")
    
    # Step 1: Load and prepare data
    print("Step 1: Loading and preparing data...")
    df, levels, level_sample_counts, df_all_agg, df_all_subagents, id_column, has_levels = \
        load_and_prepare_data(data_file, workflow_type, args)
    
    config = BENCHMARK_CONFIGS[workflow_type]
    all_level_results = {}
    
    # Step 2: Process each level
    for level in levels:
        print(f"\n{'='*60}")
        print(f"Processing Level: {level}")
        print(f"{'='*60}")
        
        # Prepare DataFrames for this level
        df_subagents, metadata = prepare_level_dataframes(
            df, level, workflow_type, config, args,
            df_all_subagents, id_column, level_sample_counts, has_levels
        )
        
        # Compute full workflow metrics
        print(f"\nComputing full workflow metrics...")
        workflow_df = compute_full_workflow_metrics(workflow_type, config, df_subagents, args)
        
        # Apply filters
        workflow_df = apply_workflow_filters(workflow_df, args)
        
        if len(workflow_df) == 0:
            print(f"⚠ Warning: No workflows remaining for level {level} after filtering. Skipping.")
            continue
        
        # Find Pareto frontier
        workflow_df, pareto_df = find_pareto_frontier(workflow_df, args)
        
        # Store results
        level_results = {
            'n_samples': metadata['n_samples'],
            'workflow_configs': workflow_df,
            'workflow_pareto': pareto_df,
            'pareto_frontier': pareto_df,  # Alias for backward compatibility
        }
        
        # Step 3: Explore workflow structures if requested
        if getattr(args, 'prune_workflow', False) and not getattr(args, 'uniform_config', False):
            print(f"\nExploring workflow structures...")
            all_structures, structure_results = explore_workflow_structures(
                workflow_type, config, df_subagents, metadata, args
            )
            
            # Combine full workflow with structure results
            # Key: Treat structure as another parameter, combine ALL configs, then find Pareto
            combined_all_configs, combined_pareto_df = combine_structure_results(
                workflow_df, structure_results, args
            )
            
            level_results['all_structures'] = all_structures
            level_results['structure_results'] = structure_results
            level_results['combined_all_configs'] = combined_all_configs
            level_results['combined_all_pareto'] = combined_pareto_df
        
        all_level_results[level] = level_results
        
        # Print summary
        print(f"\nLevel {level} Summary:")
        print(f"  Total workflows: {len(workflow_df)}")
        print(f"  Pareto optimal: {len(pareto_df)}")
        if 'combined_all_pareto' in level_results:
            print(f"  Combined Pareto (with structures): {len(combined_pareto_df)}")
    
    # Step 4: Save results and create plots
    print(f"\n{'='*60}")
    print("Saving results and creating plots...")
    print(f"{'='*60}\n")
    
    save_prediction_results(all_level_results, output_dir, args)
    create_combined_plot(all_level_results, output_dir, args)
    
    # Create per-level comparison plots if structure exploration is enabled
    if getattr(args, 'prune_workflow', False):
        create_structure_comparison_plots(all_level_results, output_dir, args)
    
    # Handle subset sampling if requested
    if getattr(args, 'sample_subset', None) is not None:
        sample_workflow_subset(all_level_results, output_dir, args)
    
    print("\n✓ Workflow prediction complete!")
    return all_level_results


def apply_workflow_filters(workflow_df, args):
    """Apply latency and accuracy filters to workflow DataFrame."""
    min_latency = getattr(args, 'min_latency', None)
    max_latency = getattr(args, 'max_latency', None)
    min_accuracy = getattr(args, 'min_accuracy', None)
    max_accuracy = getattr(args, 'max_accuracy', None)
    
    original_len = len(workflow_df)
    
    if min_latency is not None:
        workflow_df = workflow_df[workflow_df['workflow_latency'] >= min_latency]
    
    if max_latency is not None:
        workflow_df = workflow_df[workflow_df['workflow_latency'] <= max_latency]
    
    if min_accuracy is not None:
        workflow_df = workflow_df[workflow_df['workflow_accuracy'] >= min_accuracy]
    
    if max_accuracy is not None:
        workflow_df = workflow_df[workflow_df['workflow_accuracy'] <= max_accuracy]
    
    filtered_len = len(workflow_df)
    if filtered_len != original_len:
        constraints = []
        if min_latency is not None:
            constraints.append(f"min_latency>={min_latency}")
        if max_latency is not None:
            constraints.append(f"max_latency<={max_latency}")
        if min_accuracy is not None:
            constraints.append(f"min_accuracy>={min_accuracy}")
        if max_accuracy is not None:
            constraints.append(f"max_accuracy<={max_accuracy}")
        print(f"  Applied constraints ({', '.join(constraints)}): {original_len} -> {filtered_len} configs")
    
    return workflow_df


def find_pareto_frontier(workflow_df, args):
    """Find Pareto frontier in workflow DataFrame."""
    uniform_config = getattr(args, 'uniform_config', False)
    
    if uniform_config:
        print(f"Uniform config mode: Retaining all {len(workflow_df)} points (no Pareto filtering)")
        workflow_df['is_pareto'] = True
        pareto_df = workflow_df.sort_values('workflow_latency').reset_index(drop=True)
    else:
        print("Finding Pareto frontier...")
        costs = np.column_stack([
            workflow_df['workflow_latency'].values,
            -workflow_df['workflow_accuracy'].values
        ])
        
        pareto_mask = is_pareto_efficient(costs)
        workflow_df['is_pareto'] = pareto_mask
        pareto_df = workflow_df[pareto_mask].sort_values('workflow_latency').reset_index(drop=True)
        
        print(f"Pareto efficient points: {pareto_mask.sum()} out of {len(workflow_df)}")
    
    return workflow_df, pareto_df


def combine_structure_results(workflow_df, structure_results, args):
    """
    Combine full workflow with structure exploration results.
    
    Key insight: Structure is just another configuration parameter (like model choice for sub-agents).
    We combine ALL configurations from ALL structures into one pool, then find the global Pareto frontier.
    
    This is different from finding Pareto frontiers per structure and then combining them.
    """
    # Collect ALL configurations from ALL structures (not just Pareto points!)
    all_structure_configs = []
    
    for structure_id, result in structure_results.items():
        configs_df = result['configs'].copy()
        configs_df['structure_id'] = structure_id
        if 'total_branches' not in configs_df.columns:
            configs_df['total_branches'] = result['total_branches']
        all_structure_configs.append(configs_df)
        print(f"  {structure_id} ({result['total_branches']} branches): {len(configs_df):,} configs")
    
    # Add full workflow configurations  
    # The full workflow should already have structure_id and total_branches from compute step
    full_configs = workflow_df.copy()
    # Just ensure total_branches exists for logging/analysis.
    if 'total_branches' not in full_configs.columns:
        full_configs['total_branches'] = 1
    all_structure_configs.append(full_configs)
    full_structure_id = full_configs['structure_id'].iloc[0] if len(full_configs) > 0 and 'structure_id' in full_configs.columns else 'unknown'
    print(f"  {full_structure_id} ({full_configs['total_branches'].iloc[0] if len(full_configs) > 0 else 1} branches): {len(full_configs):,} configs")
    
    # Combine ALL configurations into one big pool
    combined_all_configs = pd.concat(all_structure_configs, ignore_index=True)
    print(f"\nTotal configurations across all structures: {len(combined_all_configs):,}")
    
    # Apply filters (max_latency, max_accuracy, min_accuracy) to combined configs
    original_count = len(combined_all_configs)
    combined_all_configs = apply_workflow_filters(combined_all_configs, args)
    if len(combined_all_configs) < original_count:
        print(f"After applying threshold filters: {len(combined_all_configs):,} configurations remain")
    
    # Find global Pareto frontier across ALL structures and configurations
    print("Finding global Pareto frontier across all structures and configs...")
    costs = np.column_stack([
        combined_all_configs['workflow_latency'].values,
        -combined_all_configs['workflow_accuracy'].values
    ])
    pareto_mask = is_pareto_efficient(costs)
    
    # Mark Pareto status in the combined dataframe (needed for sampling)
    combined_all_configs['is_pareto'] = pareto_mask
    combined_pareto_df = combined_all_configs[pareto_mask].sort_values('workflow_latency').reset_index(drop=True)
    
    print(f"Global Pareto frontier: {len(combined_pareto_df)} configurations")
    print(f"  Accuracy range: {combined_pareto_df['workflow_accuracy'].min():.4f} - {combined_pareto_df['workflow_accuracy'].max():.4f}")
    print(f"  Latency range: {combined_pareto_df['workflow_latency'].min():.2f}s - {combined_pareto_df['workflow_latency'].max():.2f}s")
    
    # Analyze structure distribution in Pareto frontier
    structure_counts = combined_pareto_df['structure_id'].value_counts()
    print(f"\nStructure distribution in global Pareto frontier:")
    for structure_id in sorted(structure_counts.index):
        count = structure_counts[structure_id]
        pct = (count / len(combined_pareto_df)) * 100
        branches = combined_pareto_df[combined_pareto_df['structure_id'] == structure_id]['total_branches'].iloc[0]
        print(f"  {structure_id} ({branches} branches): {count} configs ({pct:.1f}%)")
    
    return combined_all_configs, combined_pareto_df


def create_structure_comparison_plots(all_level_results, output_dir, args):
    """Create per-level comparison plots showing full vs pruned structures."""
    import matplotlib.pyplot as plt
    
    output_prefix = getattr(args, 'output_prefix', 'workflow_by_level')
    
    for level in sorted(all_level_results.keys()):
        result = all_level_results[level]
        
        if 'structure_results' not in result:
            continue
        
        plt.figure(figsize=(14, 10))
        
        # Plot full workflow Pareto
        full_pareto = result['workflow_pareto']
        if len(full_pareto) > 0:
            plt.scatter(full_pareto['workflow_latency'], full_pareto['workflow_accuracy'],
                       c='blue', marker='o', s=150, alpha=0.7, label='Full Workflow', zorder=5)
            plt.plot(full_pareto['workflow_latency'], full_pareto['workflow_accuracy'],
                    c='blue', alpha=0.4, linewidth=2, linestyle='--', zorder=4)
        
        # Plot combined Pareto (full + structures)
        if 'combined_all_pareto' in result:
            combined_pareto = result['combined_all_pareto']
            plt.scatter(combined_pareto['workflow_latency'], combined_pareto['workflow_accuracy'],
                       c='red', marker='*', s=200, alpha=0.8, edgecolors='darkred', linewidth=2,
                       label=f'Combined Pareto ({len(combined_pareto)} points)', zorder=6)
            plt.plot(combined_pareto['workflow_latency'], combined_pareto['workflow_accuracy'],
                    c='red', alpha=0.5, linewidth=3, linestyle='-', zorder=5)
        
        plt.xlabel('Workflow Latency (seconds)', fontsize=14, fontweight='bold')
        plt.ylabel('Workflow Accuracy', fontsize=14, fontweight='bold')
        plt.title(f'Level {level}: Full vs Pruned Structures', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{output_prefix}_level_{level}_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Structure comparison plots saved")


def sample_workflow_subset(all_level_results, output_dir, args):
    """Sample a subset of Pareto optimal workflows."""
    import random
    
    n_sample = getattr(args, 'sample_subset', None)
    if n_sample is None:
        return
    
    random_seed = getattr(args, 'random_seed', 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    prune_workflow = getattr(args, 'prune_workflow', False)
    
    print(f"\nSampling {n_sample} Pareto optimal configs per level...")
    
    sampled_results = {}
    for level, result in all_level_results.items():
        # Use combined Pareto when structure exploration is enabled
        if prune_workflow and 'combined_all_pareto' in result:
            pareto_configs = result['combined_all_pareto']
        elif 'workflow_pareto' in result:
            pareto_configs = result['workflow_pareto']
        else:
            # Fallback: filter from workflow_configs
            workflow_df = result.get('combined_all_configs', result.get('workflow_configs'))
            pareto_configs = workflow_df[workflow_df['is_pareto'] == True]
        
        # Sample from Pareto optimal configs only
        if len(pareto_configs) <= n_sample:
            sampled_df = pareto_configs
        else:
            sampled_df = pareto_configs.sample(n=n_sample, random_state=random_seed)
        
        sampled_results[level] = sampled_df
        
        print(f"  Level {level}: {len(sampled_df)} Pareto optimal configs")
    
    # Save sampled results
    import json
    output_prefix = getattr(args, 'output_prefix', 'workflow_by_level')
    output_file = os.path.join(output_dir, f'{output_prefix}_sampled_n{n_sample}.json')
    
    sampled_data = {
        level: df.sort_values('workflow_latency').replace({np.nan: None}).to_dict('records')
        for level, df in sampled_results.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f, indent=2)
    
    print(f"✓ Sampled results saved to {output_file}")
