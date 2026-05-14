"""Benchmark registry with auto-discovery and metadata."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type


_REGISTRY: Dict[str, Type] = {}
_CANONICAL_NAME: Dict[str, str] = {}
_ALIASES: Dict[str, List[str]] = {}
_ALIAS_TO_KEY: Dict[str, str] = {}
_DISCOVERED = False


def _normalize(name: str) -> str:
    return str(name or "").strip().lower()


def register_benchmark(name: Optional[str] = None, aliases: Optional[List[str]] = None):
    """Decorator to register a benchmark class.

    Class-level metadata defaults are filled if missing:
    - BENCHMARK_NAME
    - ALIASES
    - WORKFLOW_TYPE
    - METRIC_NAME
    - DEFAULT_SPLIT_PATHS
    """

    def decorator(benchmark_class: Type) -> Type:
        canonical_name = name or getattr(benchmark_class, "BENCHMARK_NAME", benchmark_class.__name__)
        canonical_name = str(canonical_name).strip()
        key = _normalize(canonical_name)

        class_aliases = list(getattr(benchmark_class, "ALIASES", []) or [])
        extra_aliases = list(aliases or [])
        merged_aliases: List[str] = []
        for candidate in [canonical_name, *class_aliases, *extra_aliases]:
            if not candidate:
                continue
            value = str(candidate).strip()
            if value and value not in merged_aliases:
                merged_aliases.append(value)

        workflow_type = getattr(benchmark_class, "WORKFLOW_TYPE", "math")
        metric_name = getattr(benchmark_class, "METRIC_NAME", "accuracy")
        default_split_paths = dict(getattr(benchmark_class, "DEFAULT_SPLIT_PATHS", {}) or {})
        default_init_kwargs = dict(getattr(benchmark_class, "DEFAULT_INIT_KWARGS", {}) or {})

        benchmark_class.BENCHMARK_NAME = canonical_name
        benchmark_class.ALIASES = merged_aliases
        benchmark_class.WORKFLOW_TYPE = workflow_type
        benchmark_class.METRIC_NAME = metric_name
        benchmark_class.DEFAULT_SPLIT_PATHS = default_split_paths
        benchmark_class.DEFAULT_INIT_KWARGS = default_init_kwargs

        _REGISTRY[key] = benchmark_class
        _CANONICAL_NAME[key] = canonical_name
        _ALIASES[key] = merged_aliases
        for alias in merged_aliases:
            _ALIAS_TO_KEY[_normalize(alias)] = key

        return benchmark_class

    return decorator


def discover_benchmarks(force: bool = False) -> None:
    """Auto-import benchmark modules so decorator side effects register them."""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return

    package = importlib.import_module("flowcompile.benchmarks")
    for module_info in pkgutil.iter_modules(package.__path__):
        mod = module_info.name
        if mod in {"benchmark", "registry"}:
            continue
        importlib.import_module(f"{package.__name__}.{mod}")

    _DISCOVERED = True


def get_benchmark_class(benchmark_name: str):
    discover_benchmarks()
    key = _ALIAS_TO_KEY.get(_normalize(benchmark_name))
    if key is None:
        available = ", ".join(sorted(_CANONICAL_NAME.values()))
        raise ValueError(f"Benchmark '{benchmark_name}' not found. Available benchmarks: {available}")
    return _REGISTRY[key]


def get_benchmark_info(benchmark_name: str) -> Dict[str, Any]:
    benchmark_class = get_benchmark_class(benchmark_name)
    key = _ALIAS_TO_KEY[_normalize(benchmark_name)]
    return {
        "name": _CANONICAL_NAME[key],
        "aliases": list(_ALIASES.get(key, [])),
        "workflow_type": getattr(benchmark_class, "WORKFLOW_TYPE", "math"),
        "metric_name": getattr(benchmark_class, "METRIC_NAME", "accuracy"),
        "default_split_paths": dict(getattr(benchmark_class, "DEFAULT_SPLIT_PATHS", {}) or {}),
        "default_init_kwargs": dict(getattr(benchmark_class, "DEFAULT_INIT_KWARGS", {}) or {}),
        "class": benchmark_class,
    }


def get_benchmark(benchmark_name: str, **kwargs):
    benchmark_class = get_benchmark_class(benchmark_name)
    return benchmark_class(**kwargs)


def list_benchmarks(detailed: bool = False):
    discover_benchmarks()
    keys = sorted(_REGISTRY.keys(), key=lambda k: _CANONICAL_NAME[k].lower())
    if not detailed:
        return [_CANONICAL_NAME[k] for k in keys]

    items = []
    for key in keys:
        benchmark_class = _REGISTRY[key]
        items.append(
            {
                "name": _CANONICAL_NAME[key],
                "aliases": list(_ALIASES.get(key, [])),
                "workflow_type": getattr(benchmark_class, "WORKFLOW_TYPE", "math"),
                "metric_name": getattr(benchmark_class, "METRIC_NAME", "accuracy"),
                "default_split_paths": dict(getattr(benchmark_class, "DEFAULT_SPLIT_PATHS", {}) or {}),
                "default_init_kwargs": dict(getattr(benchmark_class, "DEFAULT_INIT_KWARGS", {}) or {}),
                "class_name": benchmark_class.__name__,
                "module": benchmark_class.__module__,
            }
        )
    return items


__all__ = [
    "register_benchmark",
    "discover_benchmarks",
    "get_benchmark_class",
    "get_benchmark_info",
    "get_benchmark",
    "list_benchmarks",
]
