#!/usr/bin/env python3
"""Basic tests for the canonical KNN router."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src directory to path for direct repository test runs.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_router_import():
    from flowcompile.routers import (
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
    from flowcompile.routers import get_router, list_routers

    available = list_routers()
    assert available == ["knn"]

    router = get_router("knn", k=5, accuracy_thresholds=[0.8, 0.9])
    assert router.name == "knn"
    assert router.k == 5


def test_embedding_cache():
    from flowcompile.routers.knn import EmbeddingCache

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
    from flowcompile.routers.knn import filter_pareto_optimal, is_pareto_efficient

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


def test_router_defaults_to_longformer():
    from flowcompile.routers.knn import KNNRouter

    router = KNNRouter()
    assert router.embedding_model == "allenai/longformer-base-4096"
    assert router.max_length == 4096


def test_fit_reuses_cached_embeddings(monkeypatch, tmp_path: Path):
    from flowcompile.routers.knn import KNNRouter

    cache_file = tmp_path / "knn_embeddings.pkl"
    router = KNNRouter(embedding_cache_file=str(cache_file))
    query_table = {
        "q1": {"query_text": "query one", "agents": {"agent_a": {"s1": {"accuracy": 1.0, "latency": 1.0}}}},
        "q2": {"query_text": "query two", "agents": {"agent_a": {"s1": {"accuracy": 0.0, "latency": 2.0}}}},
    }

    embed_calls = {"count": 0}

    def fake_embed_batch(texts, batch_size=8):
        del batch_size
        embed_calls["count"] += len(texts)
        return np.array([[float(idx + 1), 0.0] for idx in range(len(texts))], dtype=float)

    monkeypatch.setattr(router.embedder, "embed_batch", fake_embed_batch)
    router.fit_from_query_table(query_table)
    assert embed_calls["count"] == 2
    assert cache_file.exists()

    router2 = KNNRouter(embedding_cache_file=str(cache_file))

    def fail_embed_batch(texts, batch_size=8):
        raise AssertionError(f"embed_batch should not run for cached texts: {texts}")

    monkeypatch.setattr(router2.embedder, "embed_batch", fail_embed_batch)
    router2.fit_from_query_table(query_table)
    assert router2.validation_embeddings.shape == (2, 2)


def test_build_runtime_candidates_uses_subset_and_full_fallback(monkeypatch):
    from flowcompile.routers.knn import KNNRouter

    class FakeWorkflowModule:
        workflow_type = "math"

        def infer_metric_agents(self):
            return ["agent_a", "agent_b"]

        def normalize_subagent_stats(self, agent_dfs):
            return {key: value.copy() for key, value in agent_dfs.items()}

        def compute_configs(self, agent_dfs, metadata):
            del metadata
            assert set(agent_dfs.keys()) == {"agent_a", "agent_b"}
            assert sorted(agent_dfs["agent_a"]["setting"].tolist()) == ["subset_only"]
            assert sorted(agent_dfs["agent_b"]["setting"].tolist()) == ["fallback_only"]
            return pd.DataFrame(
                [
                    {
                        "structure_id": "fast",
                        "agent_a_setting": "subset_only",
                        "agent_b_setting": "fallback_only",
                        "workflow_accuracy": 0.4,
                        "workflow_latency": 1.0,
                    },
                    {
                        "structure_id": "accurate",
                        "agent_a_setting": "subset_only",
                        "agent_b_setting": "fallback_only",
                        "workflow_accuracy": 0.9,
                        "workflow_latency": 3.0,
                    },
                ]
            )

    router = KNNRouter(k=1)
    router.query_data_table = {
        "subset_q": {
            "query_text": "subset query",
            "agents": {"agent_a": {"subset_only": {"accuracy": 0.8, "latency": 1.0}}},
        },
        "full_q": {
            "query_text": "full query",
            "agents": {"agent_b": {"fallback_only": {"accuracy": 0.6, "latency": 2.0}}},
        },
    }
    monkeypatch.setattr(
        router.embedder,
        "embed_batch",
        lambda texts, batch_size=8: np.array([[float(idx + 1), 0.0] for idx in range(len(texts))], dtype=float),
    )
    router._post_fit_setup()

    monkeypatch.setattr(router, "_get_neighbors", lambda query: (["subset_q"], [0.1]))
    monkeypatch.setattr("flowcompile.routers.knn.get_workflow_module", lambda workflow_type: FakeWorkflowModule())

    configs, metadata = router.build_runtime_candidates({"problem": "new query", "id": "q_new"}, "math")

    assert [cfg["config_id"] for cfg in configs] == ["knn_cfg_0000", "knn_cfg_0001"]
    assert metadata["fallback_subagents"] == ["agent_b"]
    assert configs[0]["agents"]["agent_a"]["setting"] == "subset_only"
    assert configs[0]["agents"]["agent_b"]["setting"] == "fallback_only"
