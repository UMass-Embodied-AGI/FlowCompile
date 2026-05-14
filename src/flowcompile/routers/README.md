# Routers

FlowCompile now exposes a single router implementation: `knn`.

- Router name: `knn`
- Backing implementation: KNN + Pareto frontier routing
- Removed router names: `random`, `round_robin`, `knn_pareto`

## Quick Usage

```python
from flowcompile.routers import get_router, list_routers

assert list_routers() == ["knn"]

router = get_router(
    "knn",
    k=10,
    accuracy_thresholds=[0.8, 0.85, 0.9, 0.95, 0.99],
)
```

## Runtime CLI

Router-specific runtime CLI commands are removed. Runtime execution is now unified under:

```bash
flowcompile runtime infer
```
