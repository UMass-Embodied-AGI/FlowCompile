# Router Architecture

FlowCompile uses a single router implementation: `knn`.

## Registry

- `list_routers()` returns `['knn']`
- `get_router('knn', **kwargs)` returns `KNNRouter`
- Any other router key raises `ValueError`

## Components

- `src/flowcompile/routers/__init__.py`: registry and router base types
- `src/flowcompile/routers/knn.py`: canonical KNN + Pareto router implementation
- `src/flowcompile/routers/utils.py`: consolidation and data loading helpers

## Runtime Integration

- Runtime selection + execution is exposed via `flowcompile runtime infer`.
- Router-specific runtime modules were removed from `src/flowcompile/runtime/`.
