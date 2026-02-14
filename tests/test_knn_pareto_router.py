#!/usr/bin/env python3
"""Basic tests for the canonical KNN router."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_router_import():
    from workflow_compiler.routers import (
        KNNRouter,
        Router,
        RoutingResult,
        get_router,
        list_routers,
        register_router,
    )

    assert Router is not None
    assert RoutingResult is not None
    assert register_router is not None
    assert get_router is not None
    assert list_routers() == ["knn"]
    assert KNNRouter is not None


def test_router_registry():
    from workflow_compiler.routers import get_router, list_routers

    available = list_routers()
    assert available == ["knn"]

    router = get_router("knn", k=5, accuracy_thresholds=[0.8, 0.9])
    assert router.name == "knn"
    assert router.k == 5


def test_embedding_cache():
    from workflow_compiler.routers.knn import EmbeddingCache

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        cache_file = f.name

    cache = EmbeddingCache(cache_file)
    cache.put("test1", np.array([1.0, 2.0, 3.0]))
    cache.put("test2", np.array([4.0, 5.0, 6.0]))
    cache.save()

    cache2 = EmbeddingCache(cache_file)
    assert cache2.has("test1")
    assert cache2.has("test2")

    retrieved = cache2.get("test1")
    assert retrieved is not None
    assert len(retrieved) == 3


def test_pareto_utilities():
    from workflow_compiler.routers.knn import filter_pareto_optimal, is_pareto_efficient

    costs = np.array(
        [
            [1.0, 2.0],
            [0.5, 1.0],
            [2.0, 0.5],
            [1.5, 1.5],
        ]
    )

    efficient = is_pareto_efficient(costs)
    assert bool(efficient[1])
    assert bool(efficient[2])

    df = pd.DataFrame({"accuracy": [0.8, 0.9, 0.85, 0.7], "latency": [1.0, 2.0, 1.5, 0.5]})
    pareto_df = filter_pareto_optimal(df)
    assert len(pareto_df) > 0
