"""Legacy utilities for conversion/analysis/prediction orchestration.

This module is no longer exposed through the FlowCompile CLI.
Use `flowcompile --config <yaml> run-all` for end-to-end runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flowcompile.core.analysis import (
    MODEL_TO_HF_NAME as model_to_hf_name,
    calculate_latency as wfc_calculate_latency,
    load_latency_data as wfc_load_latency_data,
)
from flowcompile.core.analysis.modeling import extract_model_size_key
from flowcompile.core.analysis.prediction import predict_workflow_by_level_simplified


# ============================================================================
# PART 1: DATA CONVERSION FUNCTIONS
# ============================================================================


def safe_model_sort_key(model_name):
    """Sort models by parsed size when possible, with stable fallback for unknown names."""
    try:
        return (0, extract_model_size_key(model_name))
    except Exception:
        return (1, model_name)


def create_problem_to_metadata_map(trace_training_data):
    """Create mapping from problem string to original sample metadata."""
    problem_to_metadata = {}
    for training_data in trace_training_data["training_data"]:
        original_sample = training_data["original_sample"]

        # HotpotQA-style traces store assembled prompt in top-level "problem".
        problem = training_data.get("problem")
        if not problem:
            problem = original_sample.get("problem")

        if problem:
            problem_to_metadata[problem] = original_sample

    return problem_to_metadata


def load_latency_data(latency_file):
    """Load latency benchmark data and create model-to-latency mapping."""
    return wfc_load_latency_data(latency_file)


def calculate_latency(input_tokens, output_tokens, model_name, model_to_io_latency_per_token):
    """Calculate latency for a given configuration."""
    return wfc_calculate_latency(input_tokens, output_tokens, model_name, model_to_io_latency_per_token)


def convert_data_to_consolidated(
    detailed_results_filename,
    trace_training_data_filename,
    latency_benchmark_filename,
    output_dir,
):
    """Convert detailed results to consolidated data with latency calculations."""
    detailed_results = {}
    if isinstance(detailed_results_filename, str):
        detailed_results_filename = [detailed_results_filename]

    for filename in detailed_results_filename:
        with open(filename, "r", encoding="utf-8") as f:
            file_data = json.load(f)
            for subagent, settings in file_data.items():
                if subagent not in detailed_results:
                    detailed_results[subagent] = {}
                for setting, entries in settings.items():
                    if setting not in detailed_results[subagent]:
                        detailed_results[subagent][setting] = []
                    detailed_results[subagent][setting].extend(entries)

    with open(trace_training_data_filename, "r", encoding="utf-8") as f:
        trace_training_data = json.load(f)

    model_to_io_latency_per_token = load_latency_data(latency_benchmark_filename)
    problem_to_metadata = create_problem_to_metadata_map(trace_training_data)

    all_data = []
    for subagent in detailed_results:
        for setting in detailed_results[subagent]:
            for entry in detailed_results[subagent][setting]:
                problem = entry["problem"]
                metadata = deepcopy(problem_to_metadata[problem])
                metadata.pop("supporting_facts", None)
                metadata.pop("context", None)
                metadata.pop("level", None)
                metadata.pop("difficulty", None)

                accuracy = entry["accuracy"]
                input_tokens = entry["avg_input_tokens"]
                output_tokens = entry["avg_output_tokens"]

                latency = calculate_latency(
                    input_tokens,
                    output_tokens,
                    setting,
                    model_to_io_latency_per_token,
                )

                saved_data = deepcopy(metadata)
                saved_data.update(
                    {
                        "subagent": subagent,
                        "setting": setting,
                        "accuracy": accuracy,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency": latency,
                    }
                )
                all_data.append(saved_data)

    df = pd.DataFrame(all_data)
    output_file = os.path.join(output_dir, "all_data.json")
    df.replace({np.nan: None}).to_json(output_file, orient="records", lines=False, indent=2)
    print(f"✓ Saved {len(df)} records to {output_file}")

    return df, output_file


# ============================================================================
# PART 2: ANALYSIS AND PLOTTING FUNCTIONS
# ============================================================================


def create_analysis_dataframe(acc_files, latency_file):
    """Create dataframe for analysis plots."""
    latency_data = json.load(open(latency_file, encoding="utf-8"))
    model_to_io_latency_per_token = {}
    for model_name, model_data in latency_data.items():
        model_data = model_data[0]
        prefill_throughput = model_data["prefill_tok_per_s"]
        decode_throughput = model_data["decode_tok_per_s"]
        model_to_io_latency_per_token[model_name] = {
            "prefill_latency_per_token": 1.0 / prefill_throughput,
            "decode_latency_per_token": 1.0 / decode_throughput,
        }

    df_table = []

    for acc_file in acc_files:
        with open(acc_file, encoding="utf-8") as f:
            acc_data = json.load(f)
        for subagent_name, subagent_data in acc_data.items():
            for cfg_name in subagent_data:
                cfg_data = subagent_data[cfg_name]
                raw_model = cfg_data["model"]
                model = model_to_hf_name.get(raw_model, raw_model)
                if model not in model_to_io_latency_per_token and raw_model in model_to_io_latency_per_token:
                    model = raw_model
                if model not in model_to_io_latency_per_token:
                    continue

                avg_input_tokens_per_sample = cfg_data["avg_input_tokens_per_sample"]
                avg_output_tokens_per_sample = cfg_data["avg_output_tokens_per_sample"]
                io_latency_per_token = model_to_io_latency_per_token[model]
                prefill_latency = avg_input_tokens_per_sample * io_latency_per_token["prefill_latency_per_token"]
                decode_latency = avg_output_tokens_per_sample * io_latency_per_token["decode_latency_per_token"]
                total_io_latency = prefill_latency + decode_latency
                overall_accuracy = cfg_data["overall_accuracy"]

                try:
                    budget_str = cfg_name.split("_")[-1]
                    if budget_str.lower() == "unlimited":
                        budget = 5000
                    else:
                        budget = float(budget_str)
                except (ValueError, IndexError):
                    budget = 0.0

                df_table.append((subagent_name, model, cfg_name, total_io_latency, overall_accuracy, budget))

    df = pd.DataFrame(
        df_table,
        columns=[
            "subagent_name",
            "model",
            "cfg_name",
            "latency",
            "accuracy",
            "budget",
        ],
    )
    return df


def generate_analysis_plots(acc_files, latency_file, output_dir, gpu_type="h100"):
    """Generate analysis plots for accuracy vs latency/budget."""
    df = create_analysis_dataframe(acc_files, latency_file)

    sorted_models = sorted(df["model"].unique(), key=safe_model_sort_key)
    colors = sns.color_palette("colorblind", len(sorted_models))
    model_colors = {model: colors[i] for i, model in enumerate(sorted_models)}

    for subagent_name in df["subagent_name"].unique():
        for column in ["latency", "budget"]:
            df_subagent = df[df["subagent_name"] == subagent_name]
            sorted_subagent_models = sorted(df_subagent["model"].unique(), key=safe_model_sort_key)
            plt.figure(figsize=(10, 6))

            sns.lineplot(
                data=df_subagent,
                x=column,
                y="accuracy",
                hue="model",
                palette=model_colors,
                hue_order=sorted_subagent_models,
                marker="o",
                linewidth=2.5,
                legend=True,
            )

            plt.title(f"Subagent: {subagent_name} - Accuracy vs {column.replace('_', ' ').title()}")
            plt.xlabel(column.replace("_", " ").title())
            plt.ylabel("Accuracy")
            plt.legend(title="Model")
            plt.grid(True)
            plt.tight_layout()
            output_file = os.path.join(output_dir, f"analyze_{subagent_name}_{column}_{gpu_type}.png")
            plt.savefig(output_file)
            plt.close()
            print(f"✓ Saved: {output_file}")


# ============================================================================
# PART 3: WORKFLOW PREDICTION
# ============================================================================


def _mode_table():
    return {
        "standard": {
            "name": "standard",
            "prune": False,
            "max_budget_only": False,
        },
        "standard-with-structures": {
            "name": "standard_with_structures",
            "prune": True,
            "max_budget_only": False,
        },
        "max-budget": {
            "name": "max_budget",
            "prune": False,
            "max_budget_only": True,
        },
    }


def auto_detect_experiment_files(experiment_id):
    """Auto-detect detailed_results, trace_data, and acc_files from experiment_id."""
    import glob

    results_dir = os.path.join("results", experiment_id)
    profile_dir = os.path.join(results_dir, "01_profile")
    data_dir = os.path.join(results_dir, "data")
    if not os.path.exists(results_dir):
        raise ValueError(f"Experiment directory not found: {results_dir}")

    benchmark_dirs = []
    for root in (profile_dir, results_dir, data_dir):
        benchmark_dirs.extend(glob.glob(os.path.join(root, "benchmark_*")))
    benchmark_dirs = sorted(set(benchmark_dirs))

    detailed_results = []
    acc_files = []
    for benchmark_dir in benchmark_dirs:
        detailed_path = os.path.join(benchmark_dir, "detailed_results.json")
        acc_path = os.path.join(benchmark_dir, "summary_statistics.json")

        if os.path.exists(detailed_path):
            detailed_results.append(detailed_path)
        if os.path.exists(acc_path):
            acc_files.append(acc_path)

    trace_data = None
    canonical_trace = os.path.join(profile_dir, "aggregated_training_data.json")
    if os.path.exists(canonical_trace):
        trace_data = canonical_trace
    else:
        trace_patterns = [
            os.path.join(data_dir, "aggregated_training_data.json"),
            os.path.join(profile_dir, "*training_data.json"),
            os.path.join(data_dir, "*training_data.json"),
            os.path.join(results_dir, "*_agent_*", "trace_training_data.json"),
            os.path.join(results_dir, "*_agent*", "trace_training_data.json"),
            os.path.join(results_dir, "*_fixed_agent_*", "trace_training_data.json"),
            os.path.join(results_dir, "*_fixed_agent*", "trace_training_data.json"),
            os.path.join(data_dir, "*_agent_*", "trace_training_data.json"),
            os.path.join(data_dir, "*_agent*", "trace_training_data.json"),
            os.path.join(data_dir, "*_fixed_agent_*", "trace_training_data.json"),
            os.path.join(data_dir, "*_fixed_agent*", "trace_training_data.json"),
        ]
        for pattern in trace_patterns:
            matches = glob.glob(pattern)
            if matches:
                trace_data = max(matches, key=os.path.getmtime)
                break

    if not detailed_results:
        raise ValueError(
            f"No detailed_results.json found in benchmark_* under {profile_dir}, {results_dir}, or {data_dir}"
        )
    if not trace_data:
        raise ValueError(
            f"No trace/aggregated training data found under {profile_dir}, {results_dir}, or {data_dir}"
        )
    if not acc_files:
        raise ValueError(
            f"No summary_statistics.json found in benchmark_* under {profile_dir}, {results_dir}, or {data_dir}"
        )

    return detailed_results, trace_data, acc_files


def run_complete_pipeline(args):
    # Normalize workflow_type: gsm8k uses math workflow structures.
    if args.workflow_type == "gsm8k":
        print("Note: gsm8k uses the same workflow as math, normalizing workflow_type to 'math'")
        args.workflow_type = "math"

    if isinstance(args.analysis_mode, str):
        args.analysis_mode = [args.analysis_mode]

    experiment_id = args.experiment_id

    if args.detailed_results is None or args.trace_data is None or args.acc_files is None:
        print(f"Auto-detecting files for experiment: {experiment_id}")
        try:
            detected_detailed, detected_trace, detected_acc = auto_detect_experiment_files(experiment_id)

            if args.detailed_results is None:
                args.detailed_results = detected_detailed
                print(f"  Detected detailed-results: {args.detailed_results}")

            if args.trace_data is None:
                args.trace_data = detected_trace
                print(f"  Detected trace-data: {args.trace_data}")

            if args.acc_files is None:
                args.acc_files = detected_acc
                print(f"  Detected acc-files: {args.acc_files}")
        except Exception as e:
            print(f"Error during auto-detection: {e}")
            print("Please provide --detailed-results, --trace-data, and --acc-files manually.")
            return 1

    compile_dir = os.path.join("results", experiment_id, args.predictions_dir_name)
    figures_dir = os.path.join(compile_dir, "figures")

    os.makedirs(compile_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if not args.skip_convert:
        _df, data_file = convert_data_to_consolidated(
            args.detailed_results,
            args.trace_data,
            args.latency_file,
            compile_dir,
        )
    else:
        data_file = os.path.join(compile_dir, "all_data.json")

    if not args.skip_analysis:
        generate_analysis_plots(
            args.acc_files,
            args.latency_file,
            figures_dir,
            args.gpu_type,
        )

    if not args.skip_prediction:
        mode_mapping = _mode_table()
        selected_modes = []

        for mode_key in args.analysis_mode:
            if mode_key not in mode_mapping:
                print(f"Skipping unknown analysis mode: {mode_key}")
                continue
            selected_modes.append(mode_mapping[mode_key])

        if not selected_modes:
            print("ERROR: No valid analysis modes selected.")
            return 1

        for mode in selected_modes:
            print(f"\n{'=' * 80}")
            print(f"Generating workflow predictions: {mode['name']}")
            print(f"{'=' * 80}")

            mode_args = argparse.Namespace(**vars(args))
            # Hard remove level-based analysis: always run all-sample mode.
            mode_args.no_split_by_level = True
            mode_args.prune_workflow = mode["prune"]
            mode_args.uniform_config = False
            mode_args.max_budget_only = mode["max_budget_only"]
            mode_args.output_prefix = mode["name"]

            predict_workflow_by_level_simplified(
                data_file,
                compile_dir,
                mode_args,
                args.workflow_type,
            )
            # Keep prediction PNGs under a dedicated figures subfolder.
            mode_png_pattern = os.path.join(compile_dir, f"{mode['name']}*.png")
            for png_path in glob.glob(mode_png_pattern):
                png_name = os.path.basename(png_path)
                target = os.path.join(figures_dir, png_name)
                if os.path.abspath(png_path) == os.path.abspath(target):
                    continue
                final_target = target
                if os.path.exists(final_target):
                    stem, ext = os.path.splitext(png_name)
                    idx = 1
                    while os.path.exists(final_target):
                        final_target = os.path.join(figures_dir, f"{stem}_{idx}{ext}")
                        idx += 1
                os.replace(png_path, final_target)

        if args.skip_synthetic:
            print("Note: --skip-synthetic is deprecated in all-sample mode and has no effect.")
    else:
        print("\nSkipping workflow prediction")

    print("\n" + "=" * 80)
    print("✓ COMPLETE PIPELINE FINISHED!")
    print("=" * 80)
    print(f"All results saved under: results/{experiment_id}/")
    print("\nDirectory structure:")
    print(f"  results/{experiment_id}/")
    print(f"  └── {args.predictions_dir_name}/ - data/predictions, with figures in {args.predictions_dir_name}/figures/")
    print("=" * 80)

    return 0
