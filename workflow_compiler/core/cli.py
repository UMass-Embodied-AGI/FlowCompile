"""FlowCompile CLI."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

import yaml
import glob
import os

from workflow_compiler.compiler.pipeline import compile_pareto
from workflow_compiler.compiler.latency import run_latency_benchmark
from workflow_compiler.compiler.ground_truth import run_ground_truth
from workflow_compiler.compiler.agent_dataset import run_agent_dataset
from workflow_compiler.compiler.profiling import run_profiling
from workflow_compiler.compiler.validation import run_validation
from workflow_compiler.runtime.infer import infer_runtime, infer_runtime_batch
from workflow_compiler.core.analysis.prediction import parse_search_axes, parse_agent_constraints
from workflow_compiler.benchmarks import get_benchmark_info
from workflow_compiler.core.data_paths import resolve_existing_path


FLAT_SCHEMA_VERSION = "flowcompile.flat.v1"
_FORBIDDEN_LEGACY_KEYS = {"compile", "validate", "runtime", "models", "global", "defaults", "shared"}
_DEFAULT_LATENCY_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]
_REQUIRED_FLAT_KEYS = {
    "schema_version",
    "experiment_id",
    "workflow_type",
    "dataset",
    "model_config",
    "validate_file",
    "test_file",
    "search_axes",
    "search_budgets",
}
_REMOVED_TEST_KEYS = {
    "test_pareto_only",
    "test_non_pareto_only",
    "test_limit",
    "test_workflow_type",
    "test_fix_empty_samples",
    "test_filter_model",
    "test_simple_workflow",
    "test_single_generate_baseline",
    "test_baseline_model",
}


def _is_flat_config(cfg: Dict[str, Any]) -> bool:
    return isinstance(cfg, dict) and cfg.get("schema_version") == FLAT_SCHEMA_VERSION


def _cfg_flat_get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(cfg, dict):
        return default
    return cfg.get(key, default)


def _load_model_config(model_config_path: str) -> Dict[str, Any]:
    with open(model_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid model config format: {model_config_path}")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise SystemExit(f"Invalid model config: missing top-level 'models' map in {model_config_path}")
    return payload


def _derive_search_models_from_latency_models(cfg: Dict[str, Any]) -> List[str]:
    latency_models = _as_list(cfg.get("latency_models")) or []
    if not latency_models:
        raise SystemExit("latency_models must be set to derive search models")

    model_config_path = cfg.get("model_config")
    if not model_config_path:
        raise SystemExit("model_config is required to derive search models from latency_models")

    payload = _load_model_config(model_config_path)
    model_entries = payload.get("models", {})
    hf_to_aliases: Dict[str, set] = {}
    for cfg_key, model_cfg in model_entries.items():
        if not isinstance(model_cfg, dict):
            continue
        hf_name = model_cfg.get("hf_model_name")
        if not hf_name:
            continue
        alias = str(model_cfg.get("model") or cfg_key).strip()
        if not alias:
            continue
        hf_to_aliases.setdefault(str(hf_name), set()).add(alias)

    derived: List[str] = []
    for hf_name in latency_models:
        aliases = sorted(hf_to_aliases.get(str(hf_name), set()))
        if not aliases:
            raise SystemExit(
                f"Unable to derive search model alias for HF model '{hf_name}'. "
                f"Add a matching entry with hf_model_name in {model_config_path}."
            )
        if len(aliases) != 1:
            raise SystemExit(
                f"Ambiguous alias mapping for HF model '{hf_name}' in {model_config_path}: {aliases}. "
                "Ensure each HF model maps to exactly one search alias."
            )
        derived.append(aliases[0])
    return derived


def _validate_flat_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise SystemExit("Config must be a YAML mapping")

    schema_version = cfg.get("schema_version")
    if schema_version != FLAT_SCHEMA_VERSION:
        raise SystemExit(
            f"Unsupported config schema_version '{schema_version}'. "
            f"Expected '{FLAT_SCHEMA_VERSION}'."
        )

    forbidden = sorted([k for k in _FORBIDDEN_LEGACY_KEYS if k in cfg])
    if forbidden:
        raise SystemExit(
            f"Unsupported nested/legacy top-level keys in flat schema: {forbidden}. "
            "Use top-level flat keys only."
        )

    missing = sorted([k for k in _REQUIRED_FLAT_KEYS if cfg.get(k) in (None, "", [])])
    if missing:
        raise SystemExit(f"Missing required flat config key(s): {missing}")

    workflow_type = str(cfg.get("workflow_type", "")).strip().lower()
    if workflow_type not in {"math", "gsm8k", "hotpotqa", "livecodebench"}:
        raise SystemExit(
            "workflow_type must be one of: math, gsm8k, hotpotqa, livecodebench"
        )

    cfg.setdefault("ground_truth_llm", "gpt-5-mini")
    cfg.setdefault("test_split", "test")
    cfg.setdefault("ground_truth_task", str(cfg.get("dataset", "")).strip().lower())

    invalid_test_keys = sorted(
        k for k in cfg.keys()
        if isinstance(k, str) and k.startswith("validate_") and k != "validate_file"
    )
    if invalid_test_keys:
        raise SystemExit(
            f"Unsupported keys for test process: {invalid_test_keys}. "
            "Use test_* keys instead."
        )

    removed_test_keys = sorted(k for k in _REMOVED_TEST_KEYS if k in cfg)
    if removed_test_keys:
        raise SystemExit(
            f"Removed test config key(s): {removed_test_keys}. "
            "Use test_pareto_sample_n and Pareto-only test mode."
        )
    if cfg.get("latency_models") in (None, "", []):
        cfg["latency_models"] = list(_DEFAULT_LATENCY_MODELS)

    latency_models = _as_list(cfg.get("latency_models"))
    if not latency_models:
        raise SystemExit("latency_models must be a non-empty list or comma-separated string")
    cfg["latency_models"] = latency_models

    cfg["search_axes"] = _as_list(cfg.get("search_axes")) or []
    cfg["search_budgets"] = _as_list(cfg.get("search_budgets")) or []
    if not cfg["search_axes"]:
        raise SystemExit("search_axes must be a non-empty list in flat config")
    if not cfg["search_budgets"]:
        raise SystemExit("search_budgets must be a non-empty list in flat config")

    for path_key in ("model_config", "validate_file", "test_file"):
        value = cfg.get(path_key)
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"{path_key} must be a non-empty string")

    return cfg


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _validate_flat_config(data)


def _cfg_get(cfg: Dict[str, Any], *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _arg_get(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _format_runtime_agent_setting(agent_info: Dict[str, Any]) -> str:
    if not isinstance(agent_info, dict):
        return "unconfigured"

    parts: List[str] = []
    setting = agent_info.get("setting")
    model = agent_info.get("model")
    budget = agent_info.get("budget")

    if setting not in (None, ""):
        parts.append(f"setting={setting}")
    if model not in (None, ""):
        parts.append(f"model={model}")
    if budget not in (None, ""):
        parts.append(f"budget={budget}")

    return ", ".join(parts) if parts else "unconfigured"


def _indent_block(text: str, prefix: str = "  ") -> str:
    lines = str(text).splitlines() or [""]
    return "\n".join(f"{prefix}{line}" for line in lines)


def _print_runtime_infer_single(result: Dict[str, Any]) -> None:
    selected_config = result.get("selected_config") or {}
    agents = selected_config.get("agents") or {}
    workflow_output = result.get("workflow_output", result.get("answer", ""))
    actual_runtime = result.get("actual_runtime_seconds")

    lines: List[str] = [
        "Used Config",
        f"  Config ID: {result.get('config_id') or selected_config.get('config_id', '')}",
        f"  Structure ID: {result.get('structure_id') or selected_config.get('structure_id', '')}",
    ]

    if agents:
        lines.append("  Sub-agents:")
        for agent_name, agent_info in agents.items():
            lines.append(f"    {agent_name}: {_format_runtime_agent_setting(agent_info)}")
    else:
        lines.append("  Sub-agents: none")

    lines.extend([
        "",
        "Workflow Output",
        _indent_block("" if workflow_output is None else str(workflow_output)),
        "",
        "Actual Runtime",
        f"  {float(actual_runtime):.3f}s" if actual_runtime is not None else "  unavailable",
        "",
        "Metadata",
        f"  Query ID: {result.get('query_id', '')}",
        f"  Output Dir: {result.get('output_dir', '')}",
    ])
    print("\n".join(lines))


def _run_correlation_experiment(args: List[str]) -> int:
    from workflow_compiler.experiments.correlation import main as correlation_main

    return correlation_main(args)


def _load_compiled_configs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("schema_version") == "flowcompile.compiled.v2":
        return data.get("configs", [])
    if isinstance(data, dict) and data.get("schema_version") == "flowcompile.compiled.v1":
        raise SystemExit(
            "Unsupported compiled schema flowcompile.compiled.v1. "
            "Recompile with `flowcompile predict` to get flowcompile.compiled.v2."
        )
    if isinstance(data, dict) and "levels" in data:
        raise SystemExit(
            "Unsupported level-based compiled config format. "
            "Recompile to flowcompile.compiled.v2 flat `configs`."
        )
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "configs" in data:
        return data.get("configs", [])
    return []


def _load_queries(path: str) -> List[Dict[str, Any]]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def _as_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return items if items else None
    return [value]


def _parse_agent_constraint_input(value: Any, kind: str) -> Dict[str, set]:
    if value is None:
        return {}
    if isinstance(value, dict):
        parsed: Dict[str, set] = {}
        for agent, vals in value.items():
            if isinstance(vals, str):
                vals_list = [v.strip() for v in vals.split(",") if v.strip()]
            elif isinstance(vals, (list, tuple, set)):
                vals_list = [str(v).strip() for v in vals if str(v).strip()]
            else:
                vals_list = [str(vals).strip()]
            parsed.update(parse_agent_constraints([f"{agent}={','.join(vals_list)}"], kind=kind))
        return parsed
    values = _as_list(value) or []
    return parse_agent_constraints(values, kind=kind)


def _build_search_space(args: Any, section_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _build_search_space_with_cfg(args, section_cfg, cfg=None, prefix=None)


def _build_search_space_with_cfg(
    args: Any,
    section_cfg: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    prefix: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    def from_cfg(key: str):
        if key in section_cfg:
            return section_cfg.get(key)
        if isinstance(cfg, dict):
            if prefix:
                prefixed_key = f"{prefix}_{key}"
                if prefixed_key in cfg:
                    return cfg.get(prefixed_key)
            if key in cfg:
                return cfg.get(key)
        return None

    raw_axes = getattr(args, "search_axes", None)
    if raw_axes is None:
        raw_axes = from_cfg("search_axes")
    axes = parse_search_axes(raw_axes)

    models = _as_list(getattr(args, "search_models", None))
    if models is None:
        models = _as_list(from_cfg("search_models"))
    if models is None and isinstance(cfg, dict) and _is_flat_config(cfg):
        models = _derive_search_models_from_latency_models(cfg)
    budgets = _as_list(getattr(args, "search_budgets", None))
    if budgets is None:
        budgets = _as_list(from_cfg("search_budgets"))
    structures = _as_list(getattr(args, "search_structures", None))
    if structures is None:
        structures = _as_list(from_cfg("search_structures"))

    agent_models_raw = getattr(args, "search_agent_models", None)
    if agent_models_raw is None:
        agent_models_raw = from_cfg("search_agent_models")
    agent_budgets_raw = getattr(args, "search_agent_budgets", None)
    if agent_budgets_raw is None:
        agent_budgets_raw = from_cfg("search_agent_budgets")

    agent_models = _parse_agent_constraint_input(agent_models_raw, kind="models")
    agent_budgets = _parse_agent_constraint_input(agent_budgets_raw, kind="budgets")

    has_constraints = any([
        models,
        budgets,
        structures,
        bool(agent_models),
        bool(agent_budgets),
        axes != {"model", "budget", "structure"},
    ])
    if not has_constraints:
        return None

    return {
        "search_axes": sorted(axes),
        "models": models,
        "budgets": budgets,
        "structures": structures,
        "agent_models": {k: sorted(v) for k, v in agent_models.items()},
        "agent_budgets": {k: sorted(v) for k, v in agent_budgets.items()},
    }


def _add_search_space_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--search-axes",
        nargs="+",
        choices=["model", "budget", "structure"],
        help="Axes to search over (default: model budget structure).",
    )
    parser.add_argument("--search-models", nargs="+", help="Global allowed models for search.")
    parser.add_argument("--search-budgets", nargs="+", help="Global allowed reasoning budgets for search.")
    parser.add_argument("--search-structures", nargs="+", help="Allowed workflow structure IDs.")
    parser.add_argument(
        "--search-agent-models",
        nargs="+",
        help="Per-agent model constraints in format agent=model1,model2",
    )
    parser.add_argument(
        "--search-agent-budgets",
        nargs="+",
        help="Per-agent budget constraints in format agent=100,200,unlimited",
    )


def _sanitize_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or "default"


def _experiment_id_from_cfg(cfg: Dict[str, Any], fallback: Optional[str] = None) -> Optional[str]:
    return (
        _cfg_flat_get(cfg, "experiment_id")
        or
        _cfg_get(cfg, "compile", "experiment_id")
        or _cfg_get(cfg, "test", "experiment_id")
        or fallback
    )


def _exp_root(experiment_id: str) -> Path:
    return Path("results") / experiment_id


def _exp_experiments_root(experiment_id: str) -> Path:
    return _exp_root(experiment_id) / "04_experiments"


def _exp_experiment_output_dir(experiment_id: str, experiment_name: str) -> str:
    return str(_exp_experiments_root(experiment_id) / experiment_name)


def _has_cli_flag(cmd_args: List[str], flag: str) -> bool:
    return any(arg == flag for arg in cmd_args)


def _resolve_canonical_latency_file(
    experiment_id: Optional[str],
    explicit_value: Optional[str] = None,
    *,
    label: str = "latency_file",
) -> str:
    if not experiment_id:
        raise SystemExit(
            f"experiment_id is required to resolve {label} at results/<experiment_id>/01_profile/latency_benchmark.json"
        )

    canonical = _exp_root(experiment_id) / "01_profile" / "latency_benchmark.json"
    canonical_str = str(canonical)

    if explicit_value and Path(explicit_value) != canonical:
        raise SystemExit(
            f"{label} must be the canonical path '{canonical_str}'. "
            "Other latency file locations are not supported."
        )

    if not canonical.exists():
        raise SystemExit(
            f"{label} not found at canonical path '{canonical_str}'. "
            "Run `flowcompile get-latency` first."
        )

    return canonical_str


def _find_latest_match(patterns: List[str], require_file: bool = True, require_dir: bool = False) -> Optional[str]:
    candidates: List[Path] = []
    for pattern in patterns:
        for matched in glob.glob(pattern):
            path = Path(matched)
            if require_file and not path.is_file():
                continue
            if require_dir and not path.is_dir():
                continue
            candidates.append(path)
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _resolve_required_input_path(
    label: str,
    explicit_value: Optional[str],
    canonical_path: Optional[str],
    detect_patterns: Optional[List[str]] = None,
) -> str:
    if explicit_value:
        return explicit_value
    if canonical_path and Path(canonical_path).exists():
        return canonical_path
    detected = _find_latest_match(detect_patterns or [], require_file=True)
    if detected:
        return detected
    if canonical_path:
        raise SystemExit(
            f"{label} not found. Checked canonical path '{canonical_path}' and auto-detection patterns {detect_patterns or []}."
        )
    raise SystemExit(f"{label} is required")


def _expand_globbed_files(paths: List[str]) -> List[str]:
    expanded: List[str] = []
    for path in paths:
        if any(ch in path for ch in ["*", "?", "["]):
            expanded.extend(glob.glob(path))
        else:
            expanded.append(path)
    return expanded


def _resolve_required_input_list(
    label: str,
    explicit_values: Optional[List[str]],
    canonical_patterns: Optional[List[str]] = None,
    fallback_patterns: Optional[List[str]] = None,
) -> List[str]:
    if explicit_values:
        resolved = _expand_globbed_files(explicit_values)
        if resolved:
            return resolved
        raise SystemExit(f"{label} patterns did not match any files: {explicit_values}")

    candidates: List[str] = []
    for pattern in (canonical_patterns or []):
        candidates.extend(glob.glob(pattern))
    if not candidates:
        latest = _find_latest_match(fallback_patterns or [], require_file=True)
        if latest:
            return [latest]
    else:
        files = sorted([p for p in candidates if Path(p).is_file()])
        if files:
            return files

    raise SystemExit(
        f"{label} not found. Checked canonical patterns {canonical_patterns or []} and fallback patterns {fallback_patterns or []}."
    )


def _benchmark_name_for_workflow(workflow_type: Optional[str]) -> str:
    workflow_type = (workflow_type or "math").lower()
    mapping = {
        "math": "MATH",
        "gsm8k": "GSM8K",
        "hotpotqa": "HotpotQA",
        "livecodebench": "LiveCodeBench",
    }
    return mapping.get(workflow_type, "MATH")


def _default_split_path_from_workflow(workflow_type: Optional[str], split: str) -> Optional[str]:
    benchmark_name = _benchmark_name_for_workflow(workflow_type)
    info = get_benchmark_info(benchmark_name)
    path = (info.get("default_split_paths") or {}).get(split)
    return resolve_existing_path(path) or path


def cmd_compile_ground_truth(args, cfg):
    gt = _cfg_get(cfg, "compile", "ground_truth", default={})
    task = (
        _arg_get(args, "task")
        or _cfg_flat_get(cfg, "ground_truth_task")
        or _cfg_flat_get(cfg, "dataset", "").lower()
        or gt.get("task")
        or "math500"
    )
    llm = _arg_get(args, "llm") or _cfg_flat_get(cfg, "ground_truth_llm") or gt.get("llm") or "gpt-5-mini"
    experiment_id = (
        _arg_get(args, "experiment_id")
        or _cfg_flat_get(cfg, "ground_truth_experiment_id")
        or _cfg_flat_get(cfg, "experiment_id")
        or gt.get("experiment_id")
        or _cfg_get(cfg, "compile", "experiment_id")
        or "default_experiment"
    )
    file_path = (
        _arg_get(args, "file_path")
        or _cfg_flat_get(cfg, "ground_truth_file")
        or _cfg_flat_get(cfg, "validate_file")
        or gt.get("file_path")
    )
    entry_point_file = (
        _arg_get(args, "entry_point_file")
        or _cfg_flat_get(cfg, "ground_truth_entry_point_file")
        or _cfg_flat_get(cfg, "livecodebench_public_test_file")
        or gt.get("entry_point_file")
    )
    file_path = resolve_existing_path(file_path) or file_path
    entry_point_file = resolve_existing_path(entry_point_file) or entry_point_file
    debug = _arg_get(args, "debug", False) or _cfg_flat_get(cfg, "ground_truth_debug", False) or gt.get("debug", False)

    gt_args = SimpleNamespace(
        task=task,
        llm=llm,
        meta_llm=_arg_get(args, "meta_llm") or _cfg_flat_get(cfg, "ground_truth_meta_llm") or gt.get("meta_llm"),
        solver_llm=_arg_get(args, "solver_llm") or _cfg_flat_get(cfg, "ground_truth_solver_llm") or gt.get("solver_llm"),
        programmer_llm=_arg_get(args, "programmer_llm") or _cfg_flat_get(cfg, "ground_truth_programmer_llm") or gt.get("programmer_llm"),
        refine_solver_llm=_arg_get(args, "refine_solver_llm") or _cfg_flat_get(cfg, "ground_truth_refine_solver_llm") or gt.get("refine_solver_llm"),
        detailed_solver_llm=_arg_get(args, "detailed_solver_llm") or _cfg_flat_get(cfg, "ground_truth_detailed_solver_llm") or gt.get("detailed_solver_llm"),
        generate_solver_llm=_arg_get(args, "generate_solver_llm") or _cfg_flat_get(cfg, "ground_truth_generate_solver_llm") or gt.get("generate_solver_llm"),
        answer_generate_llm=_arg_get(args, "answer_generate_llm") or _cfg_flat_get(cfg, "ground_truth_answer_generate_llm") or gt.get("answer_generate_llm"),
        sc_ensemble_llm=_arg_get(args, "sc_ensemble_llm") or _cfg_flat_get(cfg, "ground_truth_sc_ensemble_llm") or gt.get("sc_ensemble_llm"),
        format_answer_llm=_arg_get(args, "format_answer_llm") or _cfg_flat_get(cfg, "ground_truth_format_answer_llm") or gt.get("format_answer_llm"),
        code_generate_llm=_arg_get(args, "code_generate_llm") or _cfg_flat_get(cfg, "ground_truth_code_generate_llm") or gt.get("code_generate_llm"),
        test_llm=_arg_get(args, "test_llm") or _cfg_flat_get(cfg, "ground_truth_test_llm") or gt.get("test_llm"),
        reflection_test_llm=_arg_get(args, "reflection_test_llm") or _cfg_flat_get(cfg, "ground_truth_reflection_test_llm") or gt.get("reflection_test_llm"),
        rewriter_llm=_arg_get(args, "rewriter_llm") or _cfg_flat_get(cfg, "ground_truth_rewriter_llm") or gt.get("rewriter_llm"),
        reader_llm=_arg_get(args, "reader_llm") or _cfg_flat_get(cfg, "ground_truth_reader_llm") or gt.get("reader_llm"),
        answer_reviewer_llm=_arg_get(args, "answer_reviewer_llm") or _cfg_flat_get(cfg, "ground_truth_answer_reviewer_llm") or gt.get("answer_reviewer_llm"),
        mcp_url=(
            _arg_get(args, "mcp_url")
            or _cfg_flat_get(cfg, "ground_truth_mcp_url")
            or gt.get("mcp_url", "http://localhost:8080/mcp")
        ),
        experiment_id=experiment_id,
        debug=debug,
        file_path=file_path,
        entry_point_file=entry_point_file,
    )

    asyncio.run(run_ground_truth(gt_args))
    return 0


def cmd_compile_latency(args, cfg):
    lat = _cfg_get(cfg, "compile", "latency", default={})
    models = _arg_get(args, "models") or _cfg_flat_get(cfg, "latency_models") or lat.get("models")
    experiment_id = _experiment_id_from_cfg(cfg)
    output_json = _arg_get(args, "output_json") or _cfg_flat_get(cfg, "latency_output_json") or lat.get("output_json")
    if output_json is None and experiment_id:
        output_json = str(_exp_root(experiment_id) / "01_profile" / "latency_benchmark.json")
    if not models or not output_json:
        raise SystemExit("models and output_json are required for latency benchmarking")

    prompt_file = _arg_get(args, "prompt_file") or _cfg_flat_get(cfg, "latency_prompt_file") or lat.get("prompt_file", "data/prompts/long_text.txt")
    batch_size_arg = _arg_get(args, "batch_size")
    batch_size = batch_size_arg if batch_size_arg is not None else _cfg_flat_get(cfg, "latency_batch_size", lat.get("batch_size", 1))
    batch_sizes_arg = _arg_get(args, "batch_sizes")
    batch_sizes = batch_sizes_arg if batch_sizes_arg is not None else _cfg_flat_get(cfg, "latency_batch_sizes", lat.get("batch_sizes"))
    max_new_tokens_arg = _arg_get(args, "max_new_tokens")
    max_new_tokens = max_new_tokens_arg if max_new_tokens_arg is not None else _cfg_flat_get(cfg, "latency_max_new_tokens", lat.get("max_new_tokens", 1024))
    dtype = _arg_get(args, "dtype") or _cfg_flat_get(cfg, "latency_dtype") or lat.get("dtype", "auto")
    tp_arg = _arg_get(args, "tp")
    tp = tp_arg if tp_arg is not None else _cfg_flat_get(cfg, "latency_tp", lat.get("tp", 1))
    gpu_mem_util_arg = _arg_get(args, "gpu_mem_util")
    gpu_mem_util = gpu_mem_util_arg if gpu_mem_util_arg is not None else _cfg_flat_get(cfg, "latency_gpu_mem_util", lat.get("gpu_mem_util", 0.90))
    seed_arg = _arg_get(args, "seed")
    seed = seed_arg if seed_arg is not None else _cfg_flat_get(cfg, "latency_seed", lat.get("seed", 0))
    model_config_path = (
        _arg_get(args, "model_config_path")
        or _cfg_flat_get(cfg, "latency_model_config_path")
        or lat.get("model_config_path")
        or _cfg_flat_get(cfg, "model_config")
        or _cfg_get(cfg, "models", "config_path")
    )
    backend = _arg_get(args, "backend") or _cfg_flat_get(cfg, "latency_backend") or lat.get("backend", "openai")

    run_latency_benchmark(
        models=models,
        output_json=output_json,
        prompt_file=prompt_file,
        batch_size=batch_size,
        batch_sizes=batch_sizes,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        tp=tp,
        gpu_mem_util=gpu_mem_util,
        seed=seed,
        model_config_path=model_config_path,
        backend=backend,
    )
    return 0


def cmd_compile_prepare_data(args, cfg):
    """Run ground-truth generation and agent-dataset extraction in one step."""
    cmd_compile_ground_truth(args, cfg)
    return cmd_compile_agent_dataset(args, cfg)


def cmd_compile_agent_dataset(args, cfg):
    ad = _cfg_get(cfg, "compile", "agent_dataset", default={})
    experiment_id = _experiment_id_from_cfg(cfg)
    root = _exp_root(experiment_id) if experiment_id else None

    trace_data = _arg_get(args, "trace_data") or _cfg_flat_get(cfg, "agent_dataset_trace_data") or ad.get("trace_data")
    if trace_data is None and root is not None:
        trace_data = _resolve_required_input_path(
            "trace_data",
            explicit_value=None,
            canonical_path=_find_latest_match(
                [
                    str(root / "01_profile" / "*dsl_agent*" / "trace.jsonl"),
                    str(root / "01_profile" / "debug_dsl_*" / "trace.jsonl"),
                    str(root / "*dsl_agent*" / "trace.jsonl"),
                    str(root / "debug_dsl_*" / "trace.jsonl"),
                ],
                require_file=True,
            ),
            detect_patterns=[
                str(root / "01_profile" / "*dsl_agent*" / "trace.jsonl"),
                str(root / "01_profile" / "debug_dsl_*" / "trace.jsonl"),
                str(root / "*dsl_agent*" / "trace.jsonl"),
                str(root / "debug_dsl_*" / "trace.jsonl"),
            ],
        )
    output = _arg_get(args, "output") or _cfg_flat_get(cfg, "agent_dataset_output") or ad.get("output")
    if output is None and root is not None:
        output = str(root / "01_profile" / "aggregated_training_data.json")
    config_path = _arg_get(args, "config") or _cfg_flat_get(cfg, "agent_dataset_config") or ad.get("config") or _cfg_flat_get(cfg, "model_config") or "configs/config.yaml"
    model = _arg_get(args, "model") or _cfg_flat_get(cfg, "agent_dataset_model") or ad.get("model") or "gpt-5"
    max_samples = _arg_get(args, "max_samples")
    if max_samples is None:
        max_samples = _cfg_flat_get(cfg, "agent_dataset_max_samples", ad.get("max_samples"))
    num_workers = _arg_get(args, "num_workers")
    if num_workers is None:
        num_workers = _cfg_flat_get(cfg, "agent_dataset_num_workers", ad.get("num_workers", 20))
    individual = _arg_get(args, "individual", False) or _cfg_flat_get(cfg, "agent_dataset_individual", ad.get("individual", False))

    if not trace_data:
        raise SystemExit("trace_data is required for agent-dataset")

    run_agent_dataset(
        trace_path=trace_data,
        output=output,
        config_path=config_path,
        model=model,
        max_samples=max_samples,
        num_workers=num_workers,
        individual=individual,
    )
    return 0


def cmd_compile_profile(args, cfg):
    prof = _cfg_get(cfg, "compile", "profile", default={})
    experiment_id = (
        _arg_get(args, "experiment_id")
        or _cfg_flat_get(cfg, "profile_experiment_id")
        or _cfg_flat_get(cfg, "experiment_id")
        or prof.get("experiment_id")
        or _cfg_get(cfg, "compile", "experiment_id")
    )
    if not experiment_id:
        raise SystemExit("experiment_id is required for profiling")

    models = _arg_get(args, "models") or _cfg_flat_get(cfg, "profile_models") or prof.get("models")
    max_samples = _arg_get(args, "max_samples")
    if max_samples is None:
        max_samples = _cfg_flat_get(cfg, "profile_max_samples", prof.get("max_samples"))
    max_concurrent = _arg_get(args, "max_concurrent")
    if max_concurrent is None:
        max_concurrent = _cfg_flat_get(cfg, "profile_max_concurrent", prof.get("max_concurrent", 64))
    debug = _arg_get(args, "debug", False) or _cfg_flat_get(cfg, "profile_debug", prof.get("debug", False))
    min_samples = _arg_get(args, "min_samples_per_agent")
    if min_samples is None:
        min_samples = _cfg_flat_get(cfg, "profile_min_samples_per_agent")
    if min_samples is None:
        min_samples = _cfg_flat_get(cfg, "min_samples_per_agent")
    if min_samples is None:
        min_samples = prof.get("min_samples_per_agent", 100)
    raw_search_budgets = _arg_get(args, "search_budgets")
    if raw_search_budgets is None:
        raw_search_budgets = _cfg_flat_get(cfg, "search_budgets")
    if raw_search_budgets is None:
        raw_search_budgets = _cfg_get(cfg, "compile", "profile", "search_budgets")
    if raw_search_budgets is None:
        raw_search_budgets = prof.get("search_budgets")
    search_budgets = _as_list(raw_search_budgets)
    if search_budgets is not None:
        parsed_search_budgets: List[Any] = []
        for value in search_budgets:
            text = str(value).strip()
            if text.isdigit():
                parsed_search_budgets.append(int(text))
            else:
                parsed_search_budgets.append(text)
        search_budgets = parsed_search_budgets

    workflow_type = (
        _cfg_flat_get(cfg, "workflow_type")
        or _cfg_get(cfg, "compile", "workflow_type")
        or ""
    )
    livecodebench_validate_file = None
    livecodebench_public_test_file = None
    if str(workflow_type).lower() == "livecodebench":
        livecodebench_validate_file = _cfg_flat_get(cfg, "validate_file")
        livecodebench_public_test_file = _cfg_flat_get(cfg, "livecodebench_public_test_file")
        livecodebench_validate_file = resolve_existing_path(livecodebench_validate_file) or livecodebench_validate_file
        livecodebench_public_test_file = resolve_existing_path(livecodebench_public_test_file) or livecodebench_public_test_file

    asyncio.run(
        run_profiling(
            experiment_id=experiment_id,
            models=models,
            search_budgets=search_budgets,
            max_samples=max_samples,
            max_concurrent=max_concurrent,
            debug=debug,
            min_samples_per_agent=min_samples,
            livecodebench_validate_file=livecodebench_validate_file,
            livecodebench_public_test_file=livecodebench_public_test_file,
        )
    )
    return 0


def cmd_compile_predict(args, cfg):
    pred = _cfg_get(cfg, "compile", "predict", default={})
    experiment_id = _cfg_get(cfg, "compile", "experiment_id")
    if experiment_id is None:
        experiment_id = _cfg_flat_get(cfg, "experiment_id")
    root = _exp_root(experiment_id) if experiment_id else None
    workflow_type = (
        _arg_get(args, "workflow_type")
        or _cfg_flat_get(cfg, "predict_workflow_type")
        or pred.get("workflow_type")
        or _cfg_flat_get(cfg, "workflow_type")
        or _cfg_get(cfg, "compile", "workflow_type")
    )
    detailed_results = _arg_get(args, "detailed_results") or _cfg_flat_get(cfg, "predict_detailed_results") or pred.get("detailed_results")
    if isinstance(detailed_results, str):
        detailed_results = [detailed_results]

    trace_data = _arg_get(args, "trace_data") or _cfg_flat_get(cfg, "predict_trace_data") or pred.get("trace_data")
    latency_file = _arg_get(args, "latency_file") or _cfg_flat_get(cfg, "predict_latency_file") or pred.get("latency_file")
    output_file = _arg_get(args, "output_file") or _cfg_flat_get(cfg, "predict_output_file") or pred.get("output_file")
    plot_file = _arg_get(args, "plot_file") or _cfg_flat_get(cfg, "predict_plot_file") or pred.get("plot_file")
    include_all_arg = _arg_get(args, "include_all")
    include_all = (
        include_all_arg
        if include_all_arg is not None
        else _cfg_flat_get(cfg, "predict_include_all_configs", pred.get("include_all_configs", False))
    )
    prune_subagents_arg = _arg_get(args, "prune_subagents")
    prune_subagents = (
        prune_subagents_arg
        if prune_subagents_arg is not None
        else _cfg_flat_get(cfg, "predict_prune_subagents", pred.get("prune_subagents", True))
    )
    search_space = _build_search_space_with_cfg(args, pred, cfg=cfg, prefix="predict")

    if not workflow_type:
        raise SystemExit("workflow_type is required")

    if detailed_results:
        detailed_results = _resolve_required_input_list(
            "detailed_results",
            explicit_values=detailed_results,
        )
    elif root is not None:
        detailed_results = _resolve_required_input_list(
            "detailed_results",
            explicit_values=None,
            canonical_patterns=[
                str(root / "01_profile" / "benchmark_*" / "detailed_results.json"),
            ],
            fallback_patterns=[
                str(root / "benchmark_*" / "detailed_results.json"),
                str(root / "data" / "benchmark_*" / "detailed_results.json"),
            ],
        )
    else:
        raise SystemExit("detailed_results is required unless compile.experiment_id is set")

    trace_data = _resolve_required_input_path(
        "trace_data",
        explicit_value=trace_data,
        canonical_path=str(root / "01_profile" / "aggregated_training_data.json") if root else None,
        detect_patterns=[
            str(root / "01_profile" / "*training_data.json"),
            str(root / "data" / "*training_data.json"),
            str(root / "*dsl_agent*" / "trace_training_data.json"),
        ] if root else [],
    )
    latency_file = _resolve_canonical_latency_file(
        experiment_id,
        explicit_value=latency_file,
        label="latency_file",
    )

    if output_file is None:
        if root is None:
            raise SystemExit("output_file is required unless compile.experiment_id is set")
        output_file = str(root / "02_compile" / "compiled_configs.json")
    if plot_file is None and root is not None:
        plot_file = str(root / "02_compile" / "figures" / "compiled_latency_vs_score.png")

    compile_pareto(
        workflow_type=workflow_type,
        detailed_results=detailed_results,
        trace_data=trace_data,
        latency_file=latency_file,
        output_file=output_file,
        plot_file=plot_file,
        include_all_configs=include_all,
        search_space=search_space,
        prune_subagents=prune_subagents,
    )
    return 0


def cmd_compile_all(args, cfg):
    steps = [
        cmd_compile_latency,
        cmd_compile_prepare_data,
        cmd_compile_profile,
        cmd_compile_predict,
        cmd_test,
    ]
    for step in steps:
        result = step(args, cfg)
        if result not in (None, 0):
            return result
    return 0


def cmd_test(args, cfg):
    test_cfg = _cfg_get(cfg, "test", default={})

    def pick(name, default=None, flat_key=None):
        value = getattr(args, name, None)
        if value is not None:
            return value
        if flat_key:
            if flat_key in cfg:
                return cfg.get(flat_key)
        return test_cfg.get(name, default)

    experiment_id = (
        pick("experiment_id", flat_key="test_experiment_id")
        or _cfg_flat_get(cfg, "experiment_id")
        or _cfg_get(cfg, "compile", "experiment_id")
        or "default"
    )
    dataset = pick("dataset", _cfg_flat_get(cfg, "dataset", "MATH500"), flat_key="test_dataset")
    split = pick("split", _cfg_flat_get(cfg, "test_split", "test"), flat_key="test_split")
    root = _exp_root(experiment_id)

    config_file = pick("config_file")
    if config_file is None:
        config_file = _resolve_required_input_path(
            "config_file",
            explicit_value=None,
            canonical_path=str(root / "02_compile" / "compiled_configs.json"),
            detect_patterns=[
                str(root / "compiled" / "compiled_configs.json"),
            ],
        )

    output_dir = pick("output_dir", flat_key="test_output_dir")
    if output_dir is None:
        output_dir = str(root / "03_test")

    data_path = pick("data_path", flat_key="test_data_path")
    if data_path is None and _is_flat_config(cfg):
        if split == "test":
            data_path = _cfg_flat_get(cfg, "test_file")
        else:
            data_path = _cfg_flat_get(cfg, "validate_file")
    entry_point_file = pick("entry_point_file", flat_key="test_entry_point_file")
    if entry_point_file is None:
        entry_point_file = _cfg_flat_get(cfg, "livecodebench_public_test_file")
    entry_point_file = resolve_existing_path(entry_point_file) or entry_point_file

    ns = SimpleNamespace(
        experiment_id=experiment_id,
        config_file=config_file,
        output_dir=output_dir,
        workflow_type="fixed",
        pareto_sample_n=pick("pareto_sample_n", flat_key="test_pareto_sample_n"),
        parallel=pick("parallel", _cfg_flat_get(cfg, "test_parallel", 1), flat_key="test_parallel"),
        split=split,
        dataset=dataset,
        data_path=data_path,
        entry_point_file=entry_point_file,
        random_seed=pick("random_seed", _cfg_flat_get(cfg, "test_random_seed", 42), flat_key="test_random_seed"),
        start_idx=pick("start_idx", _cfg_flat_get(cfg, "test_start_idx", 0), flat_key="test_start_idx"),
        end_idx=pick("end_idx", flat_key="test_end_idx"),
        max_tasks=pick("max_tasks", _cfg_flat_get(cfg, "test_max_tasks", 16), flat_key="test_max_tasks"),
    )
    if ns.pareto_sample_n is not None:
        try:
            ns.pareto_sample_n = int(ns.pareto_sample_n)
        except (TypeError, ValueError) as exc:
            raise SystemExit("--pareto-sample-n must be an integer >= 1, or -1 to disable sampling") from exc
        if ns.pareto_sample_n == 0 or ns.pareto_sample_n < -1:
            raise SystemExit("--pareto-sample-n must be >= 1, or -1 to disable sampling")

    return asyncio.run(run_validation(ns))


def cmd_runtime_infer(args, cfg):
    runtime_cfg = _cfg_get(cfg, "runtime", default={})
    experiment_id = _experiment_id_from_cfg(cfg)
    root = _exp_root(experiment_id) if experiment_id else None
    workflow_type = (
        args.workflow_type
        or _cfg_flat_get(cfg, "runtime_workflow_type")
        or runtime_cfg.get("workflow_type")
        or _cfg_flat_get(cfg, "workflow_type")
        or _cfg_get(cfg, "compile", "workflow_type")
    )
    compiled = args.compiled or _cfg_flat_get(cfg, "runtime_compiled_configs") or runtime_cfg.get("compiled_configs")
    if compiled is None and root is not None:
        compiled = _resolve_required_input_path(
            "compiled",
            explicit_value=None,
            canonical_path=str(root / "02_compile" / "compiled_configs.json"),
            detect_patterns=[str(root / "compiled" / "compiled_configs.json")],
        )
    single_query = args.query
    queries_path = args.queries

    output_dir = Path(
        args.output_dir
        or _cfg_flat_get(cfg, "runtime_output_dir")
        or runtime_cfg.get("output_dir")
        or (str(root / "runtime" / "outputs") if root is not None else "runtime_outputs")
    )
    deprecated_runtime_routing_keys: List[str] = []
    deprecated_flat_runtime_keys = (
        "runtime_strategy",
        "runtime_alpha",
        "runtime_min_accuracy",
        "runtime_max_latency",
    )
    for key in deprecated_flat_runtime_keys:
        if isinstance(cfg, dict) and key in cfg:
            deprecated_runtime_routing_keys.append(key)
    if isinstance(runtime_cfg, dict):
        deprecated_nested_runtime_keys = (
            "strategy",
            "alpha",
            "min_accuracy",
            "max_latency",
        )
        for key in deprecated_nested_runtime_keys:
            if key in runtime_cfg:
                deprecated_runtime_routing_keys.append(f"runtime.{key}")
    if deprecated_runtime_routing_keys:
        keys_list = ", ".join(sorted(deprecated_runtime_routing_keys))
        raise SystemExit(
            "Runtime routing settings must be provided via CLI for `runtime infer`. "
            f"Remove YAML key(s): {keys_list}. "
            "Use `--strategy preference --alpha <value>` or "
            "`--strategy constraint --min-accuracy <value>` and/or `--max-latency <value>`."
        )

    strategy = args.strategy
    alpha = args.alpha
    min_acc = args.min_accuracy
    max_lat = args.max_latency

    if not workflow_type:
        raise SystemExit("workflow_type is required")
    if not compiled:
        raise SystemExit("compiled is required")
    if strategy is None:
        raise SystemExit("--strategy is required via CLI for runtime infer.")
    if single_query is not None and queries_path is not None:
        raise SystemExit("Provide exactly one of --query or --queries.")
    if single_query is None and queries_path is None:
        raise SystemExit("Either --query or --queries is required via CLI.")
    if strategy == "constraint":
        if alpha is not None:
            raise SystemExit("--alpha is only valid with --strategy preference.")
        if min_acc is None and max_lat is None:
            raise SystemExit("--strategy constraint requires at least one of --min-accuracy or --max-latency.")
    else:
        if alpha is None:
            raise SystemExit("--strategy preference requires --alpha.")
        if min_acc is not None or max_lat is not None:
            raise SystemExit("--min-accuracy/--max-latency are only valid with --strategy constraint.")

    output_dir.mkdir(parents=True, exist_ok=True)
    configs = _load_compiled_configs(compiled)

    if single_query:
        result = infer_runtime(
            query=single_query,
            configs=configs,
            workflow_type=workflow_type,
            output_dir=output_dir,
            strategy=strategy,
            alpha=alpha,
            min_accuracy=min_acc,
            max_latency=max_lat,
            query_id=args.query_id,
        )
        _print_runtime_infer_single(result)
        return 0

    queries = _load_queries(queries_path)
    results = infer_runtime_batch(
        queries=queries,
        configs=configs,
        workflow_type=workflow_type,
        output_dir=output_dir,
        strategy=strategy,
        alpha=alpha,
        min_accuracy=min_acc,
        max_latency=max_lat,
    )
    out_file = output_dir / "runtime_results.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    return 0

def cmd_experiments(args, cfg):
    name = args.name
    if name != "correlation":
        raise SystemExit(f"Unknown experiment script '{name}'")
    if not cfg:
        raise SystemExit(
            "flowcompile experiments requires --config with a flat YAML config file."
        )
    experiment_id = _experiment_id_from_cfg(cfg)
    if not experiment_id:
        raise SystemExit(
            "experiment_id is required in config for flowcompile experiments."
        )
    default_output_dir = _exp_experiment_output_dir(experiment_id, name)

    # Config-driven correlation mode (compile-style):
    # `flowcompile --config <yaml> experiments correlation`
    # still preserves passthrough mode when extra args are provided.
    if not (args.extra or []):
        root = _exp_root(experiment_id)

        workflow_type = _cfg_flat_get(cfg, "workflow_type")
        if not workflow_type:
            raise SystemExit(
                "workflow_type is required in config for experiments correlation."
            )

        workflow_all_results_dir = (
            _cfg_flat_get(cfg, "correlation_workflow_all_results_dir")
            or str(root / "03_test")
        )
        workflow_results_path = Path(workflow_all_results_dir)
        if not workflow_results_path.is_dir():
            raise SystemExit(
                f"workflow_all_results_dir not found: {workflow_all_results_dir}"
            )

        latency_file = _resolve_canonical_latency_file(
            experiment_id,
            explicit_value=_cfg_flat_get(cfg, "correlation_latency_file"),
            label="latency_file",
        )

        cmd_args = [
            "--workflow-all-results-dir",
            workflow_all_results_dir,
            "--workflow-type",
            str(workflow_type),
            "--latency-file",
            str(latency_file),
        ]

        output_dir = _cfg_flat_get(cfg, "correlation_output_dir") or default_output_dir
        cmd_args.extend(["--output-dir", str(output_dir)])

        if bool(_cfg_flat_get(cfg, "correlation_optimize_calibration", False)):
            cmd_args.append("--optimize-calibration")
    else:
        cmd_args = list(args.extra or [])

    # Canonical default output routing.
    if not _has_cli_flag(cmd_args, "--output-dir"):
        cmd_args.extend(["--output-dir", default_output_dir])

    return _run_correlation_experiment(cmd_args)


def main():
    parser = argparse.ArgumentParser(prog="flowcompile")
    parser.add_argument("--config", dest="flow_config", type=str, help="Path to FlowCompile YAML config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # get-latency
    lat = subparsers.add_parser("get-latency")
    lat.add_argument("--models")
    lat.add_argument("--output-json")
    lat.add_argument("--prompt-file")
    lat.add_argument("--batch-size", type=int)
    lat.add_argument("--batch-sizes")
    lat.add_argument("--max-new-tokens", type=int)
    lat.add_argument("--dtype")
    lat.add_argument("--tp", type=int)
    lat.add_argument("--gpu-mem-util", type=float)
    lat.add_argument("--seed", type=int)
    lat.add_argument("--model-config-path")
    lat.add_argument("--backend", choices=["auto", "vllm", "openai"])

    gt = subparsers.add_parser("ground-truth")
    gt.add_argument("--task")
    gt.add_argument("--llm")
    gt.add_argument("--meta-llm")
    gt.add_argument("--solver-llm")
    gt.add_argument("--programmer-llm")
    gt.add_argument("--refine-solver-llm")
    gt.add_argument("--detailed-solver-llm")
    gt.add_argument("--generate-solver-llm")
    gt.add_argument("--answer-generate-llm")
    gt.add_argument("--sc-ensemble-llm")
    gt.add_argument("--format-answer-llm")
    gt.add_argument("--code-generate-llm")
    gt.add_argument("--test-llm")
    gt.add_argument("--reflection-test-llm")
    gt.add_argument("--rewriter-llm")
    gt.add_argument("--reader-llm")
    gt.add_argument("--answer-reviewer-llm")
    gt.add_argument("--mcp-url")
    gt.add_argument("--entry-point-file")
    gt.add_argument("--experiment-id")
    gt.add_argument("--file-path")
    gt.add_argument("--debug", action="store_true")

    ad = subparsers.add_parser("agent-dataset")
    ad.add_argument("--trace-data")
    ad.add_argument("--output")
    ad.add_argument("--config")
    ad.add_argument("--model")
    ad.add_argument("--max-samples", type=int)
    ad.add_argument("--num-workers", type=int)
    ad.add_argument("--individual", action="store_true")

    prep = subparsers.add_parser("prepare-data")
    prep.add_argument("--task")
    prep.add_argument("--llm")
    prep.add_argument("--experiment-id")
    prep.add_argument("--file-path")
    prep.add_argument("--debug", action="store_true")
    prep.add_argument("--trace-data")
    prep.add_argument("--output")
    prep.add_argument("--config")
    prep.add_argument("--model")
    prep.add_argument("--max-samples", type=int)
    prep.add_argument("--num-workers", type=int)
    prep.add_argument("--individual", action="store_true")

    prof = subparsers.add_parser("profile")
    prof.add_argument("--experiment-id")
    prof.add_argument("--models", nargs="+")
    prof.add_argument("--max-samples", type=int)
    prof.add_argument("--max-concurrent", type=int)
    prof.add_argument("--debug", action="store_true")
    prof.add_argument("--min-samples-per-agent", type=int)
    prof.add_argument("--search-budgets", nargs="+")

    pred = subparsers.add_parser("predict")
    pred.add_argument("--workflow-type")
    pred.add_argument("--detailed-results", nargs="+")
    pred.add_argument("--trace-data")
    pred.add_argument("--latency-file")
    pred.add_argument("--output-file")
    pred.add_argument("--plot-file")
    pred.add_argument("--include-all", action="store_true", default=None)
    pred.add_argument("--prune-subagents", dest="prune_subagents", action="store_true", default=None)
    pred.add_argument("--no-prune-subagents", dest="prune_subagents", action="store_false")
    _add_search_space_args(pred)

    subparsers.add_parser("run-all")

    # test
    tst = subparsers.add_parser("test")
    tst.add_argument("--experiment-id")
    tst.add_argument("--config-file")
    tst.add_argument("--dataset")
    tst.add_argument("--split")
    tst.add_argument("--data-path")
    tst.add_argument("--entry-point-file")
    tst.add_argument("--output-dir")
    tst.add_argument("--pareto-sample-n", type=int)
    tst.add_argument("--parallel", type=int, default=None)
    tst.add_argument("--random-seed", type=int, default=None)
    tst.add_argument("--start-idx", type=int, dest="start_idx", default=None)
    tst.add_argument("--end-idx", type=int, dest="end_idx")
    tst.add_argument("--max-tasks", type=int, default=None)

    # runtime
    runtime = subparsers.add_parser("runtime")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)

    inf = runtime_sub.add_parser("infer")
    inf.add_argument("--query")
    inf.add_argument("--query-id")
    inf.add_argument("--compiled")
    inf.add_argument("--queries")
    inf.add_argument("--output-dir")
    inf.add_argument("--workflow-type")
    inf.add_argument("--strategy", choices=["preference", "constraint"], default=None)
    inf.add_argument("--alpha", type=float)
    inf.add_argument("--min-accuracy", type=float)
    inf.add_argument("--max-latency", type=float)

    # experiments
    exp = subparsers.add_parser("experiments")
    exp.add_argument(
        "name",
        choices=["correlation"],
    )
    exp.add_argument("extra", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg = _load_yaml(args.flow_config)

    # Optional: set model config path for LLMs
    models_config_path = _cfg_flat_get(cfg, "model_config") or _cfg_get(cfg, "models", "config_path")
    if models_config_path:
        os.environ["WORKFLOW_COMPILER_CONFIG"] = str(models_config_path)

    if args.command == "get-latency":
        return cmd_compile_latency(args, cfg)
    if args.command == "ground-truth":
        return cmd_compile_ground_truth(args, cfg)
    if args.command == "agent-dataset":
        return cmd_compile_agent_dataset(args, cfg)
    if args.command == "prepare-data":
        return cmd_compile_prepare_data(args, cfg)
    if args.command == "profile":
        return cmd_compile_profile(args, cfg)
    if args.command == "predict":
        return cmd_compile_predict(args, cfg)
    if args.command == "run-all":
        return cmd_compile_all(args, cfg)

    if args.command == "test":
        return cmd_test(args, cfg)

    if args.command == "runtime":
        if args.runtime_command == "infer":
            return cmd_runtime_infer(args, cfg)

    if args.command == "experiments":
        return cmd_experiments(args, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
