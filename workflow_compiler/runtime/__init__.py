"""FlowCompile runtime utilities."""

from __future__ import annotations

from typing import Any

__all__ = ["infer_runtime", "infer_runtime_batch"]


def infer_runtime(*args: Any, **kwargs: Any):
    from .infer import infer_runtime as _infer_runtime

    return _infer_runtime(*args, **kwargs)


def infer_runtime_batch(*args: Any, **kwargs: Any):
    from .infer import infer_runtime_batch as _infer_runtime_batch

    return _infer_runtime_batch(*args, **kwargs)
