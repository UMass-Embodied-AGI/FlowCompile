"""Packaged prompt assets used by FlowCompile benchmarks and profiling."""

from __future__ import annotations

from importlib import resources

_DEFAULT_LATENCY_PROMPT_RESOURCE = "long_text.txt"

DEFAULT_LATENCY_PROMPT_SOURCE = f"{__package__}:{_DEFAULT_LATENCY_PROMPT_RESOURCE}"
DEFAULT_LATENCY_PROMPT_TEXT = resources.files(__package__).joinpath(
    _DEFAULT_LATENCY_PROMPT_RESOURCE
).read_text(encoding="utf-8")


def get_default_latency_prompt_text() -> str:
    """Return the bundled long-form prompt used by `flowcompile get-latency`."""
    return DEFAULT_LATENCY_PROMPT_TEXT
