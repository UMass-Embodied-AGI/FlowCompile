"""Runtime configuration selection utilities."""
from __future__ import annotations

from typing import List, Dict, Any, Optional


def _get_metric(cfg: Dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = cfg.get("metrics", {})
    return float(metrics.get(key, cfg.get(key, default)))


def select_by_constraints(
    configs: List[Dict[str, Any]],
    min_accuracy: Optional[float] = None,
    max_latency: Optional[float] = None,
    prefer: str = "accuracy",
) -> Optional[Dict[str, Any]]:
    if not configs:
        return None

    filtered = []
    for cfg in configs:
        acc = _get_metric(cfg, "expected_accuracy")
        lat = _get_metric(cfg, "expected_latency")
        if min_accuracy is not None and acc < min_accuracy:
            continue
        if max_latency is not None and lat > max_latency:
            continue
        filtered.append(cfg)

    if not filtered:
        return None

    # Constraint mode picks the boundary configuration:
    # - min_accuracy set: choose the lowest accuracy that still satisfies it
    # - max_latency set: choose the highest latency that still satisfies it
    # - both set: satisfy both, then prefer lowest accuracy and highest latency
    if min_accuracy is not None and max_latency is not None:
        return min(
            filtered,
            key=lambda c: (
                _get_metric(c, "expected_accuracy"),
                -_get_metric(c, "expected_latency"),
            ),
        )
    if min_accuracy is not None:
        return min(
            filtered,
            key=lambda c: (
                _get_metric(c, "expected_accuracy"),
                -_get_metric(c, "expected_latency"),
            ),
        )
    if max_latency is not None:
        return max(
            filtered,
            key=lambda c: (
                _get_metric(c, "expected_latency"),
                -_get_metric(c, "expected_accuracy"),
            ),
        )

    # Backward-compatible fallback if no explicit constraint was provided.
    if prefer == "latency":
        return min(filtered, key=lambda c: _get_metric(c, "expected_latency"))
    return max(filtered, key=lambda c: _get_metric(c, "expected_accuracy"))


def select_by_preference(
    configs: List[Dict[str, Any]],
    budget: float,
    max_latency: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if not configs:
        return None

    latencies = [_get_metric(c, "expected_latency") for c in configs]
    if not latencies:
        return None

    if max_latency is None:
        max_latency = max(latencies) if max(latencies) > 0 else 1.0

    def utility(cfg: Dict[str, Any]) -> float:
        acc = _get_metric(cfg, "expected_accuracy")
        lat = _get_metric(cfg, "expected_latency")
        norm_latency = lat / max_latency if max_latency else 0.0
        return budget * acc + (1 - budget) * (1 - norm_latency)

    return max(configs, key=utility)


def select_config(
    configs: List[Dict[str, Any]],
    strategy: str = "preference",
    budget: float = 0.5,
    min_accuracy: Optional[float] = None,
    max_latency: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if strategy == "constraint":
        return select_by_constraints(configs, min_accuracy, max_latency)
    return select_by_preference(configs, budget, max_latency)
