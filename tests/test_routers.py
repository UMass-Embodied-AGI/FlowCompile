"""Unit tests for flowcompile.routers module."""

import pytest

from flowcompile.routers import Router, RoutingResult, get_router, list_routers


class TestRoutingResult:
    def test_routing_result_basic(self):
        ranking = [("wf1", 0.9), ("wf2", 0.7), ("wf3", 0.5)]
        result = RoutingResult(ranking=ranking)

        assert result.get_best() == "wf1"
        assert result.get_top_k(2) == ["wf1", "wf2"]
        assert len(result.ranking) == 3

    def test_routing_result_empty(self):
        result = RoutingResult(ranking=[])
        assert result.get_best() is None
        assert result.get_top_k(5) == []


class TestRouterRegistry:
    def test_list_routers(self):
        routers = list_routers()
        assert routers == ["knn"]

    def test_get_router_knn(self):
        router = get_router("knn", k=1)
        assert isinstance(router, Router)
        assert router.name == "knn"

    @pytest.mark.parametrize("router_name", ["knn_pareto", "random", "round_robin", "nonexistent_router"])
    def test_removed_or_unknown_router_rejected(self, router_name):
        with pytest.raises(ValueError, match="Router.*not found"):
            get_router(router_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
