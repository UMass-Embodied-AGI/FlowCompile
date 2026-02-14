# Routers

FlowCompile now exposes a single router implementation: `knn`.

- Router name: `knn`
- Backing implementation: KNN + Pareto frontier routing
- Removed router names: `random`, `round_robin`, `knn_pareto`

## Quick Usage

```python
from workflow_compiler.routers import get_router, list_routers

assert list_routers() == ["knn"]

router = get_router(
    "knn",
    k=10,
    accuracy_thresholds=[0.8, 0.85, 0.9, 0.95, 0.99],
)
```

## Runtime CLI

```bash
flowcompile runtime knn \
  --experiment-id my_exp \
  --workflow-type math \
  --test-data data/ours/math500_test.jsonl \
  --k 10
```

The routed output is written under `results/<experiment>/knn/`.
