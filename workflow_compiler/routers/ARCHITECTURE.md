# Router Architecture

FlowCompile uses a single router implementation: `knn`.

## Registry

- `list_routers()` returns `['knn']`
- `get_router('knn', **kwargs)` returns `KNNRouter`
- Any other router key raises `ValueError`

## Components

- `workflow_compiler/routers/__init__.py`: registry and router base types
- `workflow_compiler/routers/knn.py`: canonical KNN + Pareto router implementation
- `workflow_compiler/routers/utils.py`: consolidation and data loading helpers

## Runtime Integration

- `workflow_compiler/runtime/router.py` builds `knn` routers for runtime selection.
- `workflow_compiler/runtime/knn.py` runs batch KNN routing and output generation.
