#!/usr/bin/env python3
"""
Correlation Analysis Script

Analyzes correlation between predicted and actual metrics for workflow configurations.
Calculates Spearman correlation and pairwise agreement metrics.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from workflow_compiler.core.analysis.reporting import calculate_latency_from_trace
from workflow_compiler.core.analysis import load_latency_data
from workflow_compiler.core.llm.config import build_setting


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_output_dir_from_latency(latency_file: str) -> Path:
    latency_path = Path(latency_file)
    if latency_path.parts[-2:] == ("01_profile", "latency_benchmark.json"):
        return latency_path.parent.parent / "04_experiments" / "correlation"
    return Path("correlation_analysis")


def _agents_to_llm_configs(agents: Dict[str, Any]) -> Dict[str, str]:
    llm_configs: Dict[str, str] = {}
    for agent_name, agent_cfg in (agents or {}).items():
        if isinstance(agent_cfg, dict):
            setting = agent_cfg.get("setting")
            if not setting:
                setting = build_setting(agent_cfg.get("model"), agent_cfg.get("budget"))
        else:
            setting = str(agent_cfg) if agent_cfg is not None else None
        if setting:
            llm_configs[str(agent_name)] = str(setting)
    return llm_configs


def _extract_metrics(
    config_results: Dict[str, Any],
    config_info: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (predicted_accuracy, predicted_latency, actual_accuracy)."""
    result_metrics = config_results.get("metrics") or {}
    original_config = config_info.get("original_config") or {}
    original_metrics = original_config.get("metrics") or {}
    info_results = config_info.get("results") or {}

    predicted_accuracy = (
        config_results.get("workflow_accuracy")
        if config_results.get("workflow_accuracy") is not None
        else result_metrics.get("expected_accuracy")
    )
    if predicted_accuracy is None:
        predicted_accuracy = (
            original_config.get("workflow_accuracy")
            if original_config.get("workflow_accuracy") is not None
            else original_metrics.get("expected_accuracy")
        )

    predicted_latency = (
        config_results.get("workflow_latency")
        if config_results.get("workflow_latency") is not None
        else result_metrics.get("expected_latency")
    )
    if predicted_latency is None:
        predicted_latency = (
            original_config.get("workflow_latency")
            if original_config.get("workflow_latency") is not None
            else original_metrics.get("expected_latency")
        )

    actual_accuracy = config_results.get("actual_accuracy")
    if actual_accuracy is None:
        actual_accuracy = config_results.get("actual_f1")
    if actual_accuracy is None:
        actual_accuracy = config_results.get("actual_score")
    if actual_accuracy is None:
        actual_accuracy = info_results.get("actual_accuracy")
    if actual_accuracy is None:
        actual_accuracy = info_results.get("actual_f1")
    if actual_accuracy is None:
        actual_accuracy = info_results.get("actual_score")

    return _as_float(predicted_accuracy), _as_float(predicted_latency), _as_float(actual_accuracy)


def _extract_llm_configs(config_results: Dict[str, Any], config_info: Dict[str, Any]) -> Dict[str, str]:
    llm_configs = config_info.get("llm_configs") or {}
    if llm_configs:
        return {str(k): str(v) for k, v in llm_configs.items() if v}

    from_results = _agents_to_llm_configs(config_results.get("agents") or {})
    if from_results:
        return from_results

    original_config = config_info.get("original_config") or {}
    return _agents_to_llm_configs(original_config.get("agents") or {})


def _safe_mape(predicted: np.ndarray, actual: np.ndarray) -> float:
    """MAPE with zero-denominator protection."""
    valid = np.abs(actual) > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs((predicted[valid] - actual[valid]) / actual[valid])) * 100.0)


def calculate_pairwise_agreement(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Calculate pairwise agreement between predicted and actual rankings.
    
    Pairwise agreement measures the fraction of configuration pairs for which
    the relative ordering is correctly preserved.
    
    Formula: (1 / C(N,2)) * sum_{i<j} 1[sign(pred_i - pred_j) == sign(actual_i - actual_j)]
    
    Args:
        predicted: Array of predicted values
        actual: Array of actual values
    
    Returns:
        Pairwise agreement fraction [0, 1]
    """
    n = len(predicted)
    if n < 2:
        return np.nan
    
    agreement_count = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = predicted[i] - predicted[j]
            actual_diff = actual[i] - actual[j]
            
            # Check if signs match (both positive, both negative, or both zero)
            if np.sign(pred_diff) == np.sign(actual_diff):
                agreement_count += 1
            
            total_pairs += 1
    
    return agreement_count / total_pairs if total_pairs > 0 else np.nan


def calculate_calibrated_mae(predicted: np.ndarray, actual: np.ndarray, optimize: bool = False) -> Tuple[float, float, float]:
    """Calculate calibrated MAE using affine mapping based on two selected points.
    
    Affine mapping: y_cal = a * y_pred + b
    
    Args:
        predicted: Array of predicted values
        actual: Array of actual values
        optimize: If True, find the best two points that minimize calibrated MAE.
                 If False, use min/max points for calibration (faster).
    
    Returns:
        Tuple of (calibrated_mae, a, b) where a and b are the affine parameters
    """
    n = len(predicted)
    
    if n < 2:
        return np.nan, np.nan, np.nan
    
    if not optimize:
        # Fast mode: use min/max predicted points and their corresponding actual values
        i_min = np.argmin(predicted)
        i_max = np.argmax(predicted)
        
        pred_min = predicted[i_min]
        pred_max = predicted[i_max]
        actual_min = actual[i_min]
        actual_max = actual[i_max]
        
        # Handle edge case where all predictions are the same
        if pred_max == pred_min:
            return np.nan, np.nan, np.nan
        
        # Calculate affine parameters using the same points
        a = (actual_max - actual_min) / (pred_max - pred_min)
        b = actual_min - a * pred_min
        
        # Apply calibration
        calibrated = a * predicted + b
        
        # Calculate MAE on calibrated values
        calibrated_mae = np.mean(np.abs(calibrated - actual))
        
        return calibrated_mae, a, b
    
    else:
        # Optimization mode: try all pairs of points and find the best
        best_mae = np.inf
        best_a = np.nan
        best_b = np.nan
        
        for i in range(n):
            for j in range(i + 1, n):
                # Use points i and j for calibration
                pred_i, pred_j = predicted[i], predicted[j]
                actual_i, actual_j = actual[i], actual[j]
                
                # Skip if predictions are identical
                if pred_i == pred_j:
                    continue
                
                # Calculate affine parameters
                a = (actual_j - actual_i) / (pred_j - pred_i)
                b = actual_i - a * pred_i
                
                # Apply calibration to all points
                calibrated = a * predicted + b
                
                # Calculate MAE
                mae = np.mean(np.abs(calibrated - actual))
                
                # Update best if this is better
                if mae < best_mae:
                    best_mae = mae
                    best_a = a
                    best_b = b
        
        return best_mae, best_a, best_b


def analyze_workflow_correlation(
    workflow_all_dir: Path,
    latency_file: str,
    workflow_type: str,
    output_dir: Optional[Path] = None,
    optimize_calibration: bool = False
) -> Dict:
    """Analyze correlation between predicted and actual metrics for workflow configs.
    
    Args:
        workflow_all_dir: Directory containing workflow config_* subdirectories
        latency_file: Path to latency benchmark file
        workflow_type: 'math', 'hotpotqa', 'livecodebench', 'math500', or 'gsm8k'
        output_dir: Optional directory to save results JSON
        optimize_calibration: If True, optimize point selection for calibration. If False, use min/max.
    
    Returns:
        Dictionary with correlation metrics
    """
    workflow_all_dir = Path(workflow_all_dir)
    
    if not workflow_all_dir.exists():
        raise FileNotFoundError(f"Workflow directory not found: {workflow_all_dir}")
    
    print("\n" + "="*80)
    print("WORKFLOW CORRELATION ANALYSIS")
    print("="*80)
    print(f"Directory: {workflow_all_dir}")
    print(f"Workflow type: {workflow_type}")
    normalized_workflow_type = "math" if workflow_type in {"math500", "gsm8k"} else workflow_type
    
    # Load latency data
    latency_data = load_latency_data(latency_file)
    
    # Find all config directories
    config_dirs = sorted([d for d in workflow_all_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('config_')])
    
    if not config_dirs:
        raise FileNotFoundError(f"No config_* directories found in {workflow_all_dir}")
    
    print(f"Found {len(config_dirs)} configurations\n")
    
    # Collect data from all configs
    configs_data = []
    
    for config_dir in config_dirs:
        config_results_file = config_dir / 'config_results.json'
        trace_file = config_dir / 'trace.jsonl'
        
        if not config_results_file.exists():
            print(f"  WARNING: Skipping {config_dir.name} - no config_results.json")
            continue
        
        if not trace_file.exists():
            print(f"  WARNING: Skipping {config_dir.name} - no trace.jsonl")
            continue
        
        # Load config results
        with open(config_results_file, 'r') as f:
            config_results = json.load(f)
        
        # Load optional config_info for llm_configs/metric fallbacks.
        config_data: Dict[str, Any] = {}
        config_file = config_dir / 'config_info.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)

        predicted_accuracy, predicted_latency, actual_accuracy = _extract_metrics(
            config_results, config_data
        )
        if predicted_accuracy is None or predicted_latency is None or actual_accuracy is None:
            print(
                f"  WARNING: Skipping {config_dir.name} - missing predicted/actual metrics "
                f"(pred_acc={predicted_accuracy}, pred_lat={predicted_latency}, actual={actual_accuracy})"
            )
            continue

        llm_configs = _extract_llm_configs(config_results, config_data)
        if not llm_configs:
            print(f"  WARNING: Skipping {config_dir.name} - missing llm configs")
            continue
        
        # Calculate actual latency from trace
        latency_result = calculate_latency_from_trace(
            trace_file,
            latency_data,
            llm_configs,
            workflow_type=normalized_workflow_type,
            return_per_sample=False
        )
        
        actual_latency = latency_result['mean_latency']
        
        configs_data.append({
            'config_name': config_dir.name,
            'config_index': config_results.get('config_index'),
            'predicted_accuracy': predicted_accuracy,
            'predicted_latency': predicted_latency,
            'actual_accuracy': actual_accuracy,
            'actual_latency': actual_latency
        })
    
    if len(configs_data) < 2:
        raise ValueError(f"Need at least 2 valid configs for correlation analysis, found {len(configs_data)}")
    
    print(f"Successfully loaded {len(configs_data)} configurations\n")
    
    # Convert to arrays
    predicted_accuracy = np.array([c['predicted_accuracy'] for c in configs_data])
    predicted_latency = np.array([c['predicted_latency'] for c in configs_data])
    actual_accuracy = np.array([c['actual_accuracy'] for c in configs_data])
    actual_latency = np.array([c['actual_latency'] for c in configs_data])
    
    # Calculate correlations for accuracy
    spearman_acc, spearman_acc_p = stats.spearmanr(predicted_accuracy, actual_accuracy)
    pairwise_acc = calculate_pairwise_agreement(predicted_accuracy, actual_accuracy)
    mae_acc = np.mean(np.abs(predicted_accuracy - actual_accuracy))
    mape_acc = _safe_mape(predicted_accuracy, actual_accuracy)
    calibrated_mae_acc, a_acc, b_acc = calculate_calibrated_mae(predicted_accuracy, actual_accuracy, optimize=optimize_calibration)
    
    # Calculate correlations for latency
    spearman_lat, spearman_lat_p = stats.spearmanr(predicted_latency, actual_latency)
    pairwise_lat = calculate_pairwise_agreement(predicted_latency, actual_latency)
    mae_lat = np.mean(np.abs(predicted_latency - actual_latency))
    mape_lat = _safe_mape(predicted_latency, actual_latency)
    calibrated_mae_lat, a_lat, b_lat = calculate_calibrated_mae(predicted_latency, actual_latency, optimize=optimize_calibration)
    
    # Prepare results
    results = {
        'workflow_type': workflow_type,
        'num_configs': len(configs_data),
        'optimize_calibration': optimize_calibration,
        'accuracy_metrics': {
            'spearman_rho': round(float(spearman_acc), 4),
            'spearman_p_value': float(spearman_acc_p),
            'pairwise_agreement': round(float(pairwise_acc), 4),
            'mae': round(float(mae_acc), 4),
            'mape': float(mape_acc),
            'calibrated_mae': round(float(calibrated_mae_acc), 4) if not np.isnan(calibrated_mae_acc) else None,
            'calibration_a': round(float(a_acc), 4) if not np.isnan(a_acc) else None,
            'calibration_b': round(float(b_acc), 4) if not np.isnan(b_acc) else None
        },
        'latency_metrics': {
            'spearman_rho': round(float(spearman_lat), 4),
            'spearman_p_value': float(spearman_lat_p),
            'pairwise_agreement': round(float(pairwise_lat), 4),
            'mae': round(float(mae_lat), 4),
            'mape': float(mape_lat),
            'calibrated_mae': round(float(calibrated_mae_lat), 4) if not np.isnan(calibrated_mae_lat) else None,
            'calibration_a': round(float(a_lat), 4) if not np.isnan(a_lat) else None,
            'calibration_b': round(float(b_lat), 4) if not np.isnan(b_lat) else None
        },
        'configs': configs_data
    }
    
    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nNumber of configurations analyzed: {len(configs_data)}")
    print(f"Calibration optimization: {'ENABLED' if optimize_calibration else 'DISABLED (min/max)'}")
    print(f"\n{'-'*80}")
    print("ACCURACY CORRELATION")
    print(f"{'-'*80}")
    print(f"  Spearman ρ:         {spearman_acc:.4f} (p={spearman_acc_p:.2e})")
    print(f"  Pairwise Agreement: {pairwise_acc:.4f} ({pairwise_acc*100:.2f}%)")
    print(f"  MAE:                {mae_acc:.4f}")
    print(f"  Calibrated MAE:     {calibrated_mae_acc:.4f} (a={a_acc:.4f}, b={b_acc:.4f})")
    print(f"  MAPE:               {mape_acc:.2f}%")
    print(f"\n{'-'*80}")
    print("LATENCY CORRELATION")
    print(f"{'-'*80}")
    print(f"  Spearman ρ:         {spearman_lat:.4f} (p={spearman_lat_p:.2e})")
    print(f"  Pairwise Agreement: {pairwise_lat:.4f} ({pairwise_lat*100:.2f}%)")
    print(f"  MAE:                {mae_lat:.4f}s")
    print(f"  Calibrated MAE:     {calibrated_mae_lat:.4f}s (a={a_lat:.4f}, b={b_lat:.4f})")
    print(f"  MAPE:               {mape_lat:.2f}%")
    print(f"\n{'='*80}\n")
    
    # Save to JSON
    correlation_dir = output_dir if output_dir is not None else _default_output_dir_from_latency(latency_file)
    correlation_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = correlation_dir / f'correlation_metrics_{workflow_type}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}\n")
    
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze correlation between predicted and actual metrics for workflow configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m workflow_compiler.experiments.correlation \\
    --workflow-all-results-dir results/1221_hotpotqa/03_test \\
    --workflow-type hotpotqa

  python -m workflow_compiler.experiments.correlation \\
    --workflow-all-results-dir results/1229_math500/03_test \\
    --workflow-type math500
        """
    )
    parser.add_argument(
        "--workflow-all-results-dir",
        type=str,
        required=True,
        help="Directory containing workflow config_* subdirectories with results"
    )
    parser.add_argument(
        "--latency-file",
        type=str,
        default=None,
        help="Path to latency benchmark file (must be canonical results/<exp_name>/01_profile/latency_benchmark.json)"
    )
    parser.add_argument(
        "--workflow-type",
        type=str,
        choices=["math", "hotpotqa", "livecodebench", "math500", "gsm8k"],
        required=True,
        help="Workflow type: math, hotpotqa, livecodebench, math500, or gsm8k"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for results JSON"
    )
    parser.add_argument(
        "--optimize-calibration",
        action="store_true",
        help="Optimize point selection for calibration to minimize MAE (slower but more accurate)"
    )
    
    args = parser.parse_args(argv)
    
    # Normalize workflow_type: gsm8k uses the same workflow structure as math
    if args.workflow_type == 'gsm8k':
        print("Note: gsm8k uses the same workflow as math, normalizing workflow_type to 'math'")
        args.workflow_type = 'math'
    
    workflow_all_dir = Path(args.workflow_all_results_dir)
    if args.latency_file:
        latency_path = Path(args.latency_file)
    else:
        # Canonical layout expects workflow results under results/<exp_name>/03_test.
        if workflow_all_dir.name != "03_test":
            raise FileNotFoundError(
                "Cannot infer canonical latency file unless --workflow-all-results-dir points to "
                "results/<exp_name>/03_test. Provide --latency-file explicitly with the canonical path."
            )
        exp_root = workflow_all_dir.parent
        latency_path = exp_root / "01_profile" / "latency_benchmark.json"

    if not latency_path.exists():
        raise FileNotFoundError(
            f"Latency file not found at canonical path: {latency_path}. "
            "Run compile latency/profile first."
        )

    expected_suffix = Path("01_profile") / "latency_benchmark.json"
    if latency_path.parts[-2:] != expected_suffix.parts:
        raise ValueError(
            "Latency file must be canonical: results/<exp_name>/01_profile/latency_benchmark.json"
        )

    output_dir = Path(args.output_dir) if args.output_dir else None
    
    analyze_workflow_correlation(
        workflow_all_dir,
        str(latency_path),
        args.workflow_type,
        output_dir=output_dir,
        optimize_calibration=args.optimize_calibration
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
