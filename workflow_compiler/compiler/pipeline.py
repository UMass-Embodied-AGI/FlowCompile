"""FlowCompile compiler pipeline (Pareto configuration generation)."""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import re
import time
from datetime import datetime

import pandas as pd

from workflow_compiler.core.analysis.modeling import filter_pareto_optimal
from workflow_compiler.core.llm.config import parse_config
from workflow_compiler.core.terminal import get_reporter
from workflow_compiler.compiler.prep import convert_to_consolidated, build_subagent_stats
from workflow_compiler.routers.utils import row_to_runtime_config
from workflow_compiler.runtime.selector import (
    RUNTIME_PREFERENCE_BUDGET_PRESETS,
    select_config,
)
from workflow_compiler.workflows.dsl_registry import get_workflow_module


def _select_runtime_budget_presets(configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not configs:
        return {}
    selected: Dict[str, Dict[str, Any]] = {}
    for preset_name, budget in RUNTIME_PREFERENCE_BUDGET_PRESETS.items():
        config = select_config(
            configs,
            strategy="preference",
            budget=float(budget),
        )
        if config is not None:
            selected[preset_name] = config
    return selected


def _extract_runtime_budget_preset_plot_points(
    runtime_budget_presets: Optional[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not runtime_budget_presets:
        return []

    ordered_names = list(RUNTIME_PREFERENCE_BUDGET_PRESETS.keys())
    order_index = {name: idx for idx, name in enumerate(ordered_names)}
    by_point: Dict[Tuple[float, float], List[str]] = {}

    for preset_name, config in runtime_budget_presets.items():
        cfg = config if isinstance(config, dict) else {}
        metrics = cfg.get("metrics", {})
        raw_latency = metrics.get("expected_latency", cfg.get("expected_latency"))
        raw_accuracy = metrics.get("expected_accuracy", cfg.get("expected_accuracy"))
        if raw_latency is None or raw_accuracy is None:
            continue
        try:
            latency = float(raw_latency)
            accuracy = float(raw_accuracy)
        except (TypeError, ValueError):
            continue
        by_point.setdefault((latency, accuracy), []).append(str(preset_name))

    points: List[Dict[str, Any]] = []
    for (latency, accuracy), labels in sorted(by_point.items(), key=lambda item: (item[0][0], item[0][1])):
        labels_sorted = sorted(labels, key=lambda label: order_index.get(label, len(order_index)))
        points.append(
            {
                "latency": latency,
                "accuracy": accuracy,
                "labels": labels_sorted,
                "label_text": "/".join(labels_sorted),
            }
        )
    return points


def _apply_subagent_score_thresholds(
    df_subagents: Dict[str, pd.DataFrame],
    thresholds: Optional[Dict[str, float]],
) -> Dict[str, pd.DataFrame]:
    if not thresholds:
        return {agent: agent_df.copy() for agent, agent_df in df_subagents.items()}

    unknown = sorted(set(thresholds.keys()) - set(df_subagents.keys()))
    if unknown:
        raise ValueError(
            "Unknown subagent(s) in predict_subagent_score_thresholds: "
            f"{unknown}. Available: {sorted(df_subagents.keys())}"
        )

    filtered: Dict[str, pd.DataFrame] = {}
    for agent, agent_df in df_subagents.items():
        threshold = thresholds.get(agent)
        source = agent_df.copy()
        if threshold is None:
            filtered[agent] = source.reset_index(drop=True)
            continue
        if "accuracy" not in source.columns:
            raise ValueError(
                f"Subagent '{agent}' data is missing 'accuracy' column required for threshold filtering."
            )
        kept = source[source["accuracy"] >= float(threshold)].copy().reset_index(drop=True)
        if kept.empty:
            raise ValueError(
                f"Threshold {float(threshold):.6g} removed all settings for subagent '{agent}'."
            )
        filtered[agent] = kept
    return filtered


def _save_latency_score_plot(
    workflow_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    output_path: Path,
    workflow_type: str,
    runtime_budget_presets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    if workflow_df.empty or pareto_df.empty:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        get_reporter().child("predict").warn(f"could not render plot ({exc})")
        return None

    plot_df = workflow_df
    max_points = 50000
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=42)

    pareto_sorted = pareto_df.sort_values("workflow_latency")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.scatter(
        plot_df["workflow_latency"],
        plot_df["workflow_accuracy"],
        s=12,
        alpha=0.2,
        c="#1f77b4",
        label=f"All configs ({len(workflow_df)})",
    )
    plt.scatter(
        pareto_sorted["workflow_latency"],
        pareto_sorted["workflow_accuracy"],
        s=45,
        alpha=0.9,
        c="#d62728",
        label=f"Pareto ({len(pareto_df)})",
        zorder=3,
    )
    plt.plot(
        pareto_sorted["workflow_latency"],
        pareto_sorted["workflow_accuracy"],
        c="#d62728",
        linewidth=1.8,
        alpha=0.8,
        zorder=2,
    )
    preset_points = _extract_runtime_budget_preset_plot_points(runtime_budget_presets)
    if preset_points:
        xs = [point["latency"] for point in preset_points]
        ys = [point["accuracy"] for point in preset_points]
        plt.scatter(
            xs,
            ys,
            s=110,
            marker="X",
            c="#2ca02c",
            edgecolors="black",
            linewidths=0.8,
            label="Runtime presets",
            zorder=4,
        )
        for point in preset_points:
            plt.annotate(
                point["label_text"],
                xy=(point["latency"], point["accuracy"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "#2ca02c",
                    "alpha": 0.85,
                    "lw": 0.7,
                },
                zorder=5,
            )
    plt.xlabel("Workflow Latency (seconds)")
    plt.ylabel("Workflow Accuracy")
    plt.title(f"Workflow Accuracy vs Latency ({workflow_type})")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(output_path)


def _safe_plot_stem(name: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._")
    return stem or "subagent"


def _to_budget_sort_value(budget: Any) -> float:
    if budget is None:
        return float("inf")
    if isinstance(budget, (int, float)):
        return float(budget)
    text = str(budget).strip().lower()
    if text == "nothinking":
        return 0.0
    if text == "unlimited":
        return float("inf")
    try:
        return float(text)
    except Exception:
        return float("inf")


def _extract_plot_model_budget(setting: Any) -> tuple[str, float]:
    setting_str = str(setting)
    try:
        model, budget = parse_config(setting_str)
        model_name = model or setting_str
        return model_name, _to_budget_sort_value(budget)
    except Exception:
        return setting_str, float("inf")


def _save_subagent_latency_score_plots(
    df_subagents: Dict[str, pd.DataFrame],
    output_dir: Path,
    workflow_type: str,
) -> Dict[str, str]:
    if not df_subagents:
        return {}

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        get_reporter().child("predict").warn(f"could not render per-subagent plots ({exc})")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    for subagent, sub_df in sorted(df_subagents.items()):
        if sub_df.empty or "latency" not in sub_df.columns or "accuracy" not in sub_df.columns:
            continue

        try:
            plot_df = sub_df.copy()
            if "setting" in plot_df.columns:
                model_values: List[str] = []
                budget_values: List[float] = []
                for setting in plot_df["setting"]:
                    model_name, budget_value = _extract_plot_model_budget(setting)
                    model_values.append(model_name)
                    budget_values.append(budget_value)
                plot_df["model"] = model_values
                plot_df["budget_sort"] = budget_values
            else:
                plot_df["model"] = "unknown"
                plot_df["budget_sort"] = float("inf")

            out_path = output_dir / f"analyze_{_safe_plot_stem(subagent)}_latency_h100.png"
            plt.figure(figsize=(10, 7))
            model_names = sorted({str(m) for m in plot_df["model"].dropna().tolist()})
            cmap = plt.get_cmap("tab20")
            for idx, model_name in enumerate(model_names):
                df_model = plot_df[plot_df["model"] == model_name].sort_values(
                    ["budget_sort", "latency"],
                    ascending=[True, True],
                )
                if df_model.empty:
                    continue
                plt.plot(
                    df_model["latency"],
                    df_model["accuracy"],
                    marker="o",
                    linewidth=2.2,
                    markersize=5,
                    alpha=0.95,
                    label=model_name,
                    color=cmap(idx % 20),
                )

            plt.xlabel("Latency")
            plt.ylabel("Score")
            plt.title(f"Subagent: {subagent} - Score vs Latency ({workflow_type})")
            plt.grid(True, alpha=0.3, linestyle="--")
            plt.legend(title="Model", loc="best", framealpha=0.9)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            written[subagent] = str(out_path)
        except Exception as exc:
            get_reporter().child("predict").warn(
                f"failed to plot subagent '{subagent}' ({exc})"
            )
            try:
                plt.close()
            except Exception:
                pass

    return written


def _row_to_runtime_config(row: pd.Series, workflow_type: str, workflow_module, config_id: str) -> Dict[str, Any]:
    del workflow_module
    return row_to_runtime_config(row, workflow_type, config_id)


def _build_compiled_configs(
    workflow_df: pd.DataFrame,
    workflow_type: str,
    workflow_module,
    include_all: bool,
    show_progress: bool = True,
) -> Dict[str, Any]:
    if workflow_df.empty:
        return {"configs": []}
    reporter = get_reporter().child("predict")

    pareto_df = filter_pareto_optimal(
        workflow_df,
        accuracy_col="workflow_accuracy",
        latency_col="workflow_latency",
    )
    pareto_df = pareto_df.sort_values(
        ["workflow_latency", "workflow_accuracy"],
        ascending=[True, False],
    ).reset_index(drop=True)
    pareto_df["is_pareto"] = True
    pareto_df["pareto_rank"] = pareto_df.index + 1

    configs: List[Dict[str, Any]] = []
    pareto_iter = pareto_df.iterrows()
    if show_progress:
        pareto_iter = reporter.progress(
            pareto_iter,
            total=len(pareto_df),
            desc="Formatting Pareto configs",
            unit="cfg",
            leave=False,
        )
    for idx, row in pareto_iter:
        config_id = f"cfg_{idx:04d}"
        configs.append(_row_to_runtime_config(row, workflow_type, workflow_module, config_id))

    out: Dict[str, Any] = {"configs": configs}
    if include_all:
        all_configs: List[Dict[str, Any]] = []
        workflow_df_sorted = workflow_df.sort_values(
            ["workflow_latency", "workflow_accuracy"],
            ascending=[True, False],
        ).reset_index(drop=True)
        all_iter = workflow_df_sorted.iterrows()
        if show_progress:
            all_iter = reporter.progress(
                all_iter,
                total=len(workflow_df_sorted),
                desc="Formatting all configs",
                unit="cfg",
                leave=False,
            )
        for idx, row in all_iter:
            config_id = f"all_{idx:06d}"
            row = row.copy()
            row["is_pareto"] = False
            row["pareto_rank"] = 0
            all_configs.append(_row_to_runtime_config(row, workflow_type, workflow_module, config_id))
        out["all_configs"] = all_configs
    return out


def compile_pareto(
    workflow_type: str,
    detailed_results: List[str],
    trace_data: str,
    latency_file: str,
    output_file: str,
    plot_file: Optional[str] = None,
    include_all_configs: bool = False,
    search_space: Optional[Dict[str, Any]] = None,
    prune_subagents: bool = True,
    subagent_score_thresholds: Optional[Dict[str, float]] = None,
    openclaw_lobster_workflow_file: Optional[str] = None,
    workflow_loops: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compile Pareto-optimal workflow configurations.

    Args:
        workflow_type: math | hotpotqa | livecodebench | gsm8k
        detailed_results: list of detailed_results.json files
        trace_data: trace_training_data.json file
        latency_file: latency benchmark JSON
        output_file: where to write compiled configs JSON
        plot_file: optional path for latency-vs-score plot PNG
        include_all_configs: whether to include non-Pareto configs
        search_space: optional model/budget/structure constraints for search
        prune_subagents: whether to Pareto-prune subagent settings before Cartesian expansion
        subagent_score_thresholds: optional per-subagent minimum score thresholds
    """
    # Normalize workflow_type aliases
    if workflow_type.lower() in ["math500", "math-500"]:
        workflow_type = "math"
    workflow_type = workflow_type.lower()
    workflow_module = get_workflow_module(
        workflow_type,
        openclaw_lobster_workflow_file=openclaw_lobster_workflow_file,
    )
    reporter = get_reporter().child("predict")

    start_time = time.perf_counter()
    reporter.step("Loading inputs")

    df, _ = convert_to_consolidated(
        detailed_results,
        trace_data,
        latency_file,
        show_progress=False,
    )
    if df.empty:
        raise ValueError("No consolidated data produced. Check inputs.")

    compiled: Dict[str, Any] = {
        "schema_version": "flowcompile.compiled.v2",
        "workflow_type": workflow_type,
        "generated_at": datetime.now().isoformat(),
        "metadata": {
            "detailed_results": detailed_results,
            "trace_data": trace_data,
            "latency_file": latency_file,
            "search_space": search_space,
            "prune_subagents": bool(prune_subagents and not include_all_configs),
            "subagent_score_thresholds": dict(subagent_score_thresholds or {}),
            "openclaw_lobster_workflow_file": openclaw_lobster_workflow_file,
            "workflow_loops": workflow_loops,
        },
        "configs": [],
    }

    reporter.step("Aggregating sub-agent stats")
    raw_df_subagents = build_subagent_stats(df)
    normalized_subagents = workflow_module.normalize_subagent_stats(raw_df_subagents)
    threshold_before_counts = {agent: len(agent_df) for agent, agent_df in normalized_subagents.items()}
    df_subagents = _apply_subagent_score_thresholds(normalized_subagents, subagent_score_thresholds)
    threshold_after_counts = {agent: len(agent_df) for agent, agent_df in df_subagents.items()}
    if threshold_before_counts and subagent_score_thresholds:
        reporter.detail(
            "subagent configs "
            f"{sum(threshold_before_counts.values())} -> {sum(threshold_after_counts.values())} "
            "(score thresholds)"
        )
    compiled["metadata"]["subagent_counts_before_threshold"] = threshold_before_counts
    compiled["metadata"]["subagent_counts_after_threshold"] = threshold_after_counts

    pre_counts = dict(threshold_after_counts)
    should_prune = bool(prune_subagents and not include_all_configs)
    if should_prune:
        pruned_subagents: Dict[str, pd.DataFrame] = {}
        for agent, agent_df in df_subagents.items():
            pruned_subagents[agent] = filter_pareto_optimal(
                agent_df,
                accuracy_col="accuracy",
                latency_col="latency",
            )
        df_subagents = pruned_subagents
    post_counts = {agent: len(agent_df) for agent, agent_df in df_subagents.items()}
    if pre_counts:
        reporter.detail(
            "subagent configs "
            f"{sum(pre_counts.values())} -> {sum(post_counts.values())} "
            f"(prune_subagents={should_prune})"
        )
    compiled["metadata"]["subagent_counts_before_prune"] = pre_counts
    compiled["metadata"]["subagent_counts_after_prune"] = post_counts

    metadata = {
        "search_space": search_space,
        "workflow_loops": workflow_loops,
        "show_progress": True,
        "progress_desc": "compile predict configs",
    }
    reporter.step("Generating workflow configs")
    workflow_df = workflow_module.compute_configs(df_subagents, metadata)

    resolved = workflow_df.attrs.get("search_space_resolved")
    if resolved is not None:
        compiled["metadata"]["search_space_resolved"] = resolved

    reporter.step("Building compiled config payload")
    pareto_df = pd.DataFrame()
    runtime_budget_presets: Dict[str, Dict[str, Any]] = {}
    if workflow_df.empty:
        compiled["configs"] = []
        compiled["runtime_budget_presets"] = runtime_budget_presets
    else:
        pareto_df = filter_pareto_optimal(
            workflow_df,
            accuracy_col="workflow_accuracy",
            latency_col="workflow_latency",
        ).sort_values(
            ["workflow_latency", "workflow_accuracy"],
            ascending=[True, False],
        ).reset_index(drop=True)
        compiled_result = _build_compiled_configs(
            workflow_df,
            workflow_type,
            workflow_module,
            include_all_configs,
            show_progress=False,
        )
        compiled["configs"] = compiled_result["configs"]
        runtime_budget_presets = _select_runtime_budget_presets(compiled["configs"])
        compiled["runtime_budget_presets"] = runtime_budget_presets
        if include_all_configs:
            compiled["all_configs"] = compiled_result.get("all_configs", [])

    reporter.step("Writing compiled output")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2)

    plot_path = Path(plot_file) if plot_file else output_path.with_name(f"{output_path.stem}_latency_vs_score.png")
    plot_written = _save_latency_score_plot(
        workflow_df,
        pareto_df,
        plot_path,
        workflow_type,
        runtime_budget_presets=runtime_budget_presets,
    )
    subagent_plot_files = _save_subagent_latency_score_plots(
        raw_df_subagents,
        output_path.parent / "figures",
        workflow_type,
    )
    updated_metadata = False
    if plot_written:
        compiled["metadata"]["plot_file"] = plot_written
        reporter.detail(f"Plot saved: {plot_written}")
        updated_metadata = True
    if subagent_plot_files:
        compiled["metadata"]["subagent_score_latency_plots"] = subagent_plot_files
        reporter.detail(
            "Per-subagent plots saved: "
            f"{len(subagent_plot_files)} -> {output_path.parent / 'figures'}"
        )
        updated_metadata = True
    if updated_metadata:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(compiled, f, indent=2)

    elapsed = time.perf_counter() - start_time
    compiled["metadata"]["source_record_count"] = len(df)
    compiled["metadata"]["workflow_candidate_count"] = len(workflow_df)
    compiled["metadata"]["elapsed_seconds"] = elapsed
    reporter.success(
        f"Predict finished in {elapsed:.2f}s"
    )

    return compiled
