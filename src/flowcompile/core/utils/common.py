"""Common utility helpers for IO, retries, and small data helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from pydantic_core import to_jsonable_python

T = TypeVar("T")


# =============================================================================
# IO helpers
# =============================================================================


def read_json_file(json_file: str, encoding: str = "utf-8") -> Any:
    path = Path(json_file)
    if not path.exists():
        raise FileNotFoundError(f"json_file: {json_file} not exist, return []")
    with open(path, "r", encoding=encoding) as fin:
        try:
            return json.load(fin)
        except Exception as exc:
            raise ValueError(f"read json file: {json_file} failed") from exc


def write_json_file(json_file: str, data: Any, encoding: str = "utf-8", indent: int = 4):
    folder_path = Path(json_file).parent
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=to_jsonable_python)


def read_jsonl_file(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl_file(path: str, records: List[Dict[str, Any]], encoding: str = "utf-8"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# Retry decorators
# =============================================================================


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Async retry decorator with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logging.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
            raise last_exception

        return wrapper

    return decorator


def retry_sync(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Synchronous retry decorator with exponential backoff."""

    import time

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logging.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
            raise last_exception

        return wrapper

    return decorator


# =============================================================================
# Misc helpers
# =============================================================================


def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Setup a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_list(nested_list: List[List[T]]) -> List[T]:
    return [item for sublist in nested_list for item in sublist]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


__all__ = [
    "read_json_file",
    "write_json_file",
    "read_jsonl_file",
    "write_jsonl_file",
    "retry_async",
    "retry_sync",
    "setup_logger",
    "ensure_list",
    "flatten_list",
    "safe_divide",
]
