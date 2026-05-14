"""Router subsystem for FlowCompile."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of a routing decision."""

    ranking: List[Tuple[str, float]]
    metadata: Optional[Dict[str, Any]] = None

    def get_top_k(self, k: int = 1) -> List[str]:
        return [wf_id for wf_id, _ in self.ranking[:k]]

    def get_best(self) -> Optional[str]:
        if self.ranking:
            return self.ranking[0][0]
        return None


class Router(ABC):
    """Abstract base class for routers."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

    @abstractmethod
    def route(
        self,
        query: Dict[str, Any],
        candidate_workflows: List[Dict[str, Any]],
        top_k: int = 1,
        **kwargs,
    ) -> RoutingResult:
        raise NotImplementedError

    def fit(self, training_data: List[Dict[str, Any]], **kwargs):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


_ROUTER_REGISTRY: Dict[str, Callable[..., Router]] = {}


def register_router(name: str):
    def decorator(router_class: type) -> type:
        if not issubclass(router_class, Router):
            raise TypeError(f"Router class must inherit from Router: {router_class}")
        _ROUTER_REGISTRY[name] = router_class
        logger.debug("Registered router: %s -> %s", name, router_class.__name__)
        return router_class

    return decorator


def get_router(name: str, **kwargs) -> Router:
    if name not in _ROUTER_REGISTRY:
        available = ", ".join(_ROUTER_REGISTRY.keys())
        raise ValueError(f"Router '{name}' not found. Available routers: {available}")
    return _ROUTER_REGISTRY[name](name=name, **kwargs)


def list_routers() -> List[str]:
    return list(_ROUTER_REGISTRY.keys())


from .knn import KNNRouter

__all__ = [
    "Router",
    "RoutingResult",
    "register_router",
    "get_router",
    "list_routers",
    "KNNRouter",
]
