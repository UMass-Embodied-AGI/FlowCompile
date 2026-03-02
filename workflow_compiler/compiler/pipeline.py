"""FlowCompile compiler pipeline (Pareto configuration generation)."""
from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from workflow_compiler.core.analysis.modeling import filter_pareto_optimal
from workflow_compiler.compiler.prep import convert_to_consolidated, build_subagent_stats
from workflow_compiler.routers.utils import row_to_runtime_config
from workflow_compiler.workflows.dsl_registry import get_workflow_module


def _save_latency_score_plot(
    workflow_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    output_path: Path,
    workflow_type: str,
) -> Optional[str]:
    if workflow_df.empty or pareto_df.empty:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[compile predict] warning: could not render plot ({exc})")
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
        print(f"[compile predict] warning: could not render per-subagent plots ({exc})")
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
                for setting in plot_df["setting"]:
                    setting_str = str(setting)
                    try:
                        model, _budget = parse_config(setting_str)
                    except Exception:
                        model = setting_str
                    model_values.append(model or setting_str)
                plot_df["model"] = model_values
            else:
                plot_df["model"] = "unknown"

            out_path = output_dir / f"analyze_{_safe_plot_stem(subagent)}_latency_h100.png"
            plt.figure(figsize=(10, 7))
            model_names = sorted({str(m) for m in plot_df["model"].dropna().tolist()})
            cmap = plt.get_cmap("tab20")
            for idx, model_name in enumerate(model_names):
                df_model = plot_df[plot_df["model"] == model_name].sort_values("latency")
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
            print(f"[compile predict] warning: failed to plot subagent '{subagent}' ({exc})")
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
        pareto_iter = tqdm(
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
            all_iter = tqdm(
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
    """
    # Normalize workflow_type aliases
    if workflow_type.lower() in ["math500", "math-500"]:
        workflow_type = "math"
    workflow_type = workflow_type.lower()
    workflow_module = get_workflow_module(workflow_type)

    start_time = time.perf_counter()
    print("[compile predict] loading inputs")

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
        },
        "configs": [],
    }

    print("[compile predict] aggregating subagent stats")
    raw_df_subagents = build_subagent_stats(df)
    df_subagents = workflow_module.normalize_subagent_stats(raw_df_subagents)
    pre_counts = {agent: len(agent_df) for agent, agent_df in df_subagents.items()}
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
        print(
            "[compile predict] subagent configs "
            f"{sum(pre_counts.values())} -> {sum(post_counts.values())} "
            f"(prune_subagents={should_prune})"
        )
    compiled["metadata"]["subagent_counts_before_prune"] = pre_counts
    compiled["metadata"]["subagent_counts_after_prune"] = post_counts

    metadata = {
        "search_space": search_space,
        "show_progress": True,
        "progress_desc": "compile predict configs",
    }
    print("[compile predict] generating workflow configs")
    workflow_df = workflow_module.compute_configs(df_subagents, metadata)

    resolved = workflow_df.attrs.get("search_space_resolved")
    if resolved is not None:
        compiled["metadata"]["search_space_resolved"] = resolved

    print("[compile predict] building compiled config payload")
    pareto_df = pd.DataFrame()
    if workflow_df.empty:
        compiled["configs"] = []
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
        if include_all_configs:
            compiled["all_configs"] = compiled_result.get("all_configs", [])

    print("[compile predict] writing output")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2)

    plot_path = Path(plot_file) if plot_file else output_path.with_name(f"{output_path.stem}_latency_vs_score.png")
    plot_written = _save_latency_score_plot(workflow_df, pareto_df, plot_path, workflow_type)
    subagent_plot_files = _save_subagent_latency_score_plots(
        raw_df_subagents,
        output_path.parent / "figures",
        workflow_type,
    )
    updated_metadata = False
    if plot_written:
        compiled["metadata"]["plot_file"] = plot_written
        print(f"[compile predict] plot saved: {plot_written}")
        updated_metadata = True
    if subagent_plot_files:
        compiled["metadata"]["subagent_score_latency_plots"] = subagent_plot_files
        print(
            "[compile predict] per-subagent plots saved: "
            f"{len(subagent_plot_files)} -> {output_path.parent / 'figures'}"
        )
        updated_metadata = True
    if updated_metadata:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(compiled, f, indent=2)

    elapsed = time.perf_counter() - start_time
    print(
        f"[compile predict] done in {elapsed:.2f}s | "
        f"records={len(df)} | workflow_candidates={len(workflow_df)} | "
        f"pareto_configs={len(compiled.get('configs', []))}"
    )

    return compiled
