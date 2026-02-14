"""KNN routing utilities for FlowCompile."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm

from workflow_compiler.routers import get_router, RoutingResult
from workflow_compiler.routers.utils import (
    consolidate_validation_data,
    load_test_queries,
    save_consolidated_data,
    load_consolidated_data,
)


def _default_data_files(workflow_type: str) -> Optional[List[str]]:
    data_map = {
        "math": [
            "data/ours/math_test.jsonl",
            "data/ours/math_validate.jsonl",
        ],
        "gsm8k": [
            "data/ours/gsm8k_test.jsonl",
            "data/ours/gsm8k_validate.jsonl",
        ],
        "hotpotqa": [
            "data/ours/hotpotqa_test.jsonl",
            "data/ours/hotpotqa_validate.jsonl",
        ],
        "livecodebench": [
            "data/ours/livecodebench_test.jsonl",
            "data/ours/livecodebench_validate.jsonl",
        ],
    }
    return data_map.get(workflow_type)


def save_results(
    results: Dict[str, Dict[str, Any]],
    output_file: str,
    workflow_type: str,
    query_data_table: Optional[Dict[str, Any]] = None,
    accuracy_thresholds: Optional[List[float]] = None,
    test_data_file: Optional[str] = None,
    id_key_name: Optional[str] = None,
) -> None:
    """Save routing results to JSON file in step5-compatible format."""
    if accuracy_thresholds is None:
        accuracy_thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

    output_data: Dict[str, Any] = {}

    metadata = {
        "workflow_type": workflow_type,
        "test_data_file": test_data_file,
        "dataset": workflow_type.upper() if workflow_type != "livecodebench" else "LiveCodeBench",
        "id_key_name": id_key_name,
        "saved_timestamp": datetime.now().isoformat(),
    }
    output_data["_metadata"] = metadata

    for query_id, result in results.items():
        routing_result = result["routing_result"]
        query_text = result["query_text"]

        pareto_configs = routing_result.metadata.get("pareto_configs", []) if routing_result.metadata else []

        configurations = []
        for config in pareto_configs:
            config_dict = {
                "question_id": query_id,
                "expected_accuracy": config["expected_accuracy"],
                "expected_latency": config["expected_latency"],
                "accuracy_threshold": config.get("accuracy_threshold"),
                "structure_id": config.get("structure_id", ""),
                "subagent_settings": config["subagent_settings"],
                "config_string": config["workflow_id"],
            }
            config_dict.update(config["workflow_params"])
            configurations.append(config_dict)

        neighbor_ids = routing_result.metadata.get("neighbor_ids", []) if routing_result.metadata else []
        if query_data_table:
            top_k_similar_queries = [
                {
                    "id": neighbor_id,
                    "query": query_data_table.get(neighbor_id, {}).get("query_text"),
                }
                for neighbor_id in neighbor_ids
            ]
        else:
            top_k_similar_queries = [{"id": nid} for nid in neighbor_ids]

        output_data[query_id] = {
            "query": query_text,
            "top_k_similar_queries": top_k_similar_queries,
            "configurations": configurations,
        }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def auto_detect_experiment_files(experiment_id: str) -> Tuple[List[str], str, List[str]]:
    """Auto-detect detailed_results, trace_data, and acc_files from experiment_id."""
    import glob

    results_dir = os.path.join("results", experiment_id)
    profile_dir = os.path.join(results_dir, "01_profile")
    data_dir = os.path.join(results_dir, "data")

    if not os.path.exists(results_dir):
        raise ValueError(f"Experiment directory not found: {results_dir}")

    benchmark_dirs: List[str] = []
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
    canonical_training = os.path.join(profile_dir, "aggregated_training_data.json")
    if os.path.exists(canonical_training):
        trace_data = canonical_training
    else:
        trace_patterns = [
            os.path.join(data_dir, "aggregated_training_data.json"),
            os.path.join(profile_dir, "*training_data.json"),
            os.path.join(data_dir, "*training_data.json"),
            os.path.join(results_dir, "*_agent_*", "trace_training_data.json"),
            os.path.join(results_dir, "*_fixed_agent_*", "trace_training_data.json"),
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
            f"No aggregated/trace training data found under {profile_dir}, {results_dir}, or {data_dir}"
        )
    if not acc_files:
        raise ValueError(
            f"No summary_statistics.json found in benchmark_* under {profile_dir}, {results_dir}, or {data_dir}"
        )

    return detailed_results, trace_data, acc_files


def run_knn(args) -> int:
    """Run KNN-based Pareto routing and save per-query configurations."""
    if not args.experiment_id:
        raise SystemExit("experiment_id is required")
    if not args.workflow_type:
        raise SystemExit("workflow_type is required")
    if not args.test_data:
        defaults = _default_data_files(args.workflow_type)
        if defaults:
            args.test_data = defaults[0]
    if not args.test_data:
        raise SystemExit("test_data is required")

    if getattr(args, "k", None) is None:
        args.k = 10
    if getattr(args, "embedding_model", None) is None:
        args.embedding_model = "allenai/longformer-base-4096"
    if getattr(args, "max_length", None) is None:
        args.max_length = 4096
    if getattr(args, "batch_size", None) is None:
        args.batch_size = 8
    canonical_latency = Path("results") / args.experiment_id / "01_profile" / "latency_benchmark.json"
    explicit_latency = getattr(args, "latency_file", None)
    if explicit_latency and Path(explicit_latency) != canonical_latency:
        raise SystemExit(
            f"latency_file must be the canonical path '{canonical_latency}'. "
            "Other latency file locations are not supported."
        )
    if not canonical_latency.exists():
        raise SystemExit(
            f"latency_file not found at canonical path '{canonical_latency}'. "
            "Run `flowcompile get-latency` first."
        )
    args.latency_file = str(canonical_latency)
    if not getattr(args, "accuracy_thresholds", None):
        args.accuracy_thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

    detailed_results = args.detailed_results
    trace_data = args.trace_data

    if not detailed_results or not trace_data:
        detected_detailed, detected_trace, _ = auto_detect_experiment_files(args.experiment_id)
        if not detailed_results:
            detailed_results = detected_detailed
        if not trace_data:
            trace_data = detected_trace

    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / args.experiment_id / "knn"
    output_dir.mkdir(parents=True, exist_ok=True)

    consolidated_file = output_dir / "validation_data_consolidated.json"

    if args.use_cached_consolidation and consolidated_file.exists():
        query_data_table = load_consolidated_data(str(consolidated_file))
    else:
        data_files = args.data_files
        if isinstance(data_files, str):
            data_files = [data_files]
        if data_files is None:
            data_files = _default_data_files(args.workflow_type)
        if data_files:
            data_files = [path for path in data_files if Path(path).exists()]
            if not data_files:
                data_files = None

        consolidation_workflow_type = "math" if args.workflow_type == "gsm8k" else args.workflow_type
        query_data_table = consolidate_validation_data(
            detailed_results,
            trace_data,
            args.latency_file,
            consolidation_workflow_type,
            data_files=data_files,
        )
        save_consolidated_data(query_data_table, str(consolidated_file))

    embedding_cache_file = args.embedding_cache_file or str(output_dir / "embedding_cache.pkl")

    router = get_router(
        "knn",
        k=args.k,
        embedding_model=args.embedding_model,
        max_length=args.max_length,
        embedding_cache_file=embedding_cache_file,
        accuracy_thresholds=args.accuracy_thresholds,
        embedding_batch_size=args.batch_size,
        search_space=getattr(args, "search_space", None),
    )

    router.fit_from_query_table(query_data_table)

    test_queries, id_key_name = load_test_queries(args.test_data, args.workflow_type)
    routing_workflow_type = "math" if args.workflow_type == "gsm8k" else args.workflow_type

    results: Dict[str, Dict[str, Any]] = {}
    for test_query in tqdm(test_queries, desc="Processing test queries"):
        query_id = test_query["query_id"]
        query_text = test_query["query_text"]
        try:
            routing_result = router.route(
                query={"query_text": query_text, "query_id": query_id},
                workflow_type=routing_workflow_type,
                top_k=len(args.accuracy_thresholds),
            )
            results[query_id] = {
                "query_text": query_text,
                "routing_result": routing_result,
            }
        except Exception:
            results[query_id] = {
                "query_text": query_text,
                "routing_result": RoutingResult(ranking=[], metadata={}),
            }

    output_file = output_dir / f"{args.workflow_type}_knn_k{args.k}.json"
    save_results(
        results,
        str(output_file),
        args.workflow_type,
        query_data_table=query_data_table,
        accuracy_thresholds=args.accuracy_thresholds,
        test_data_file=args.test_data,
        id_key_name=id_key_name,
    )

    return 0
