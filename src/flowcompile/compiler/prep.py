"""Data preparation utilities for FlowCompile compiler."""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from flowcompile.core.analysis import load_latency_data, calculate_latency


def _load_detailed_results(
    detailed_results_files: List[str],
    show_progress: bool = True,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    detailed_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    file_iter = detailed_results_files
    if show_progress:
        file_iter = tqdm(
            detailed_results_files,
            desc="Loading detailed results",
            unit="file",
            leave=False,
        )

    for filename in file_iter:
        with open(filename, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        for subagent, settings in file_data.items():
            if subagent not in detailed_results:
                detailed_results[subagent] = {}
            for setting, entries in settings.items():
                if setting not in detailed_results[subagent]:
                    detailed_results[subagent][setting] = []
                detailed_results[subagent][setting].extend(entries)
    return detailed_results


def _create_problem_to_metadata_map(trace_training_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    problem_to_metadata: Dict[str, Dict[str, Any]] = {}
    for training_data in trace_training_data.get("training_data", []):
        original_sample = training_data.get("original_sample", {})
        problem = training_data.get("problem") or original_sample.get("problem")
        if problem:
            problem_to_metadata[problem] = original_sample
    return problem_to_metadata


def convert_to_consolidated(
    detailed_results_files: List[str],
    trace_training_data_file: str,
    latency_file: str,
    output_file: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Convert profiling + trace data into consolidated records.

    Returns DataFrame and optionally writes JSON if output_file is provided.
    """
    detailed_results = _load_detailed_results(detailed_results_files, show_progress=show_progress)

    with open(trace_training_data_file, "r", encoding="utf-8") as f:
        trace_training_data = json.load(f)

    latency_data = load_latency_data(latency_file)
    problem_to_metadata = _create_problem_to_metadata_map(trace_training_data)

    total_records = sum(
        len(entries)
        for settings in detailed_results.values()
        for entries in settings.values()
    )

    records: List[Dict[str, Any]] = []
    progress_bar = None
    if show_progress and total_records > 0:
        progress_bar = tqdm(
            total=total_records,
            desc="Consolidating profiling entries",
            unit="entry",
            leave=False,
        )

    for subagent, settings in detailed_results.items():
        for setting, entries in settings.items():
            for entry in entries:
                problem = entry.get("problem")
                metadata = problem_to_metadata.get(problem, {})

                # Remove large fields if present
                if "supporting_facts" in metadata:
                    metadata = dict(metadata)
                    metadata.pop("supporting_facts", None)
                if "context" in metadata:
                    metadata = dict(metadata)
                    metadata.pop("context", None)
                if "level" in metadata or "difficulty" in metadata:
                    metadata = dict(metadata)
                    metadata.pop("level", None)
                    metadata.pop("difficulty", None)

                accuracy = entry.get("accuracy", 0.0)
                input_tokens = entry.get("avg_input_tokens", 0)
                output_tokens = entry.get("avg_output_tokens", 0)

                latency = calculate_latency(
                    input_tokens,
                    output_tokens,
                    setting,
                    latency_data,
                )

                record = dict(metadata)
                if problem and "problem" not in record:
                    record["problem"] = problem
                record.update({
                    "subagent": subagent,
                    "setting": setting,
                    "accuracy": accuracy,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency": latency,
                })
                records.append(record)
                if progress_bar is not None:
                    progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    df = pd.DataFrame(records)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        return df, str(output_path)

    return df, None


def build_subagent_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Aggregate consolidated data into per-agent DataFrames."""
    if df.empty:
        return {}

    agg = df.groupby(["subagent", "setting"]).agg({
        "accuracy": "mean",
        "latency": "mean",
        "input_tokens": "mean",
        "output_tokens": "mean",
    }).reset_index()

    subagents = {}
    for subagent in agg["subagent"].unique():
        sub_df = agg[agg["subagent"] == subagent].copy().reset_index(drop=True)
        subagents[subagent] = sub_df
    return subagents
