"""LLM and experiment configuration helpers."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml


MODEL_CONFIG_JSON_ENV = "WORKFLOW_COMPILER_CONFIG_JSON"
_DEFAULT_MODEL_CONFIG_OVERRIDE: Optional[Dict[str, Any]] = None


class ThinkingBudgetLLM:
    """Wrapper around AsyncLLM that routes calls through thinking-budget APIs when configured."""

    def __init__(self, base_llm, thinking_budget):
        self.base_llm = base_llm
        self.thinking_budget = thinking_budget
        self.usage_tracker = base_llm.usage_tracker
        self.config = base_llm.config

    async def __call__(self, prompt, return_io_tokens=False):
        if self.thinking_budget is not None:
            return await self.base_llm.call_with_thinking_budget(
                prompt, self.thinking_budget, return_io_tokens=return_io_tokens
            )
        return await self.base_llm(prompt, return_io_tokens=return_io_tokens)

    def get_usage_summary(self):
        return self.base_llm.get_usage_summary()

    async def call_with_thinking_budget(self, prompt, budget, return_io_tokens=False):
        return await self.base_llm.call_with_thinking_budget(prompt, budget, return_io_tokens=return_io_tokens)

    async def aclose(self):
        await self.base_llm.aclose()


# =============================================================================
# LLM setting-string helpers
# =============================================================================


def parse_config(cfg_string: str) -> Tuple[Optional[str], Any]:
    """Parse `model_budget_x` string into `(model, budget)`.

    Budget can be `int`, `'unlimited'`, `'nothinking'`, or `None`.
    """
    if cfg_string is None:
        return None, None

    if "_budget_" in cfg_string:
        parts = cfg_string.split("_budget_", 1)
        model_name = parts[0]
        budget_str = parts[1]
        if budget_str == "unlimited":
            budget = "unlimited"
        elif budget_str == "nothinking":
            budget = "nothinking"
        else:
            try:
                budget = int(budget_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid budget value: {budget_str}. Must be integer, 'unlimited', or 'nothinking'"
                ) from exc
        return model_name, budget

    return cfg_string, None


def build_setting(model: Optional[str], budget: Optional[Any]) -> Optional[str]:
    """Build canonical `model_budget_x` setting string from model/budget fields."""
    if not model:
        return None
    if budget is None:
        return model
    if isinstance(budget, str):
        if budget in {"unlimited", "nothinking"}:
            return f"{model}_budget_{budget}"
        return f"{model}_budget_{budget}"
    if isinstance(budget, (int, float)) and int(budget) == -1:
        return f"{model}_budget_unlimited"
    return f"{model}_budget_{budget}"


# =============================================================================
# Config dataclasses / file loading
# =============================================================================


@dataclass
class LLMConfig:
    """Configuration for a single LLM."""

    model: str
    budget: Optional[int] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    benchmark: str
    workflow_type: str
    output_dir: str
    llm_configs: Dict[str, Any] = field(default_factory=dict)
    router_config: Optional[Dict[str, Any]] = None
    eval_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=data.get("name", "experiment"),
            benchmark=data.get("benchmark", ""),
            workflow_type=data.get("workflow_type", ""),
            output_dir=data.get("output_dir", "results"),
            llm_configs=data.get("llm_configs", {}),
            router_config=data.get("router", None),
            eval_config=data.get("eval_config", {}),
        )


def get_config_path(config_path: Optional[str] = None) -> Path:
    """Get configuration path from arg/env/defaults."""
    if config_path:
        return Path(config_path)

    env_config = os.environ.get("WORKFLOW_COMPILER_CONFIG")
    if env_config:
        return Path(env_config)

    candidates = [Path("configs/config.yaml"), Path("configs/config2.yaml")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_model_config_payload(payload: Any, source: str = "model_config") -> Dict[str, Any]:
    """Validate and normalize a model-config payload."""
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid model config format in {source}")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Invalid model config: missing top-level 'models' map in {source}")
    return payload


def set_default_model_config_payload(payload: Optional[Dict[str, Any]]) -> None:
    """Set or clear the in-memory default model-config payload."""
    global _DEFAULT_MODEL_CONFIG_OVERRIDE
    _DEFAULT_MODEL_CONFIG_OVERRIDE = copy.deepcopy(payload) if payload is not None else None


def get_default_model_config_payload() -> Optional[Dict[str, Any]]:
    """Return the in-memory default model-config payload if one is set."""
    if _DEFAULT_MODEL_CONFIG_OVERRIDE is None:
        return None
    return copy.deepcopy(_DEFAULT_MODEL_CONFIG_OVERRIDE)


def serialize_model_config_payload(payload: Dict[str, Any]) -> str:
    """Serialize a validated model-config payload for env propagation."""
    normalized = validate_model_config_payload(copy.deepcopy(payload))
    return json.dumps(normalized, sort_keys=True)


def _load_model_config_from_path(config_path: Union[str, Path], *, base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    path = Path(config_path)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return validate_model_config_payload(payload, source=str(path))


def load_model_config_payload(
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load model-config payload from mapping, path, env, or defaults."""
    if isinstance(config, dict):
        return validate_model_config_payload(copy.deepcopy(config))
    if isinstance(config, (str, Path)):
        return _load_model_config_from_path(config, base_dir=base_dir)

    default_payload = get_default_model_config_payload()
    if default_payload is not None:
        return default_payload

    env_payload = os.environ.get(MODEL_CONFIG_JSON_ENV)
    if env_payload:
        try:
            parsed = json.loads(env_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {MODEL_CONFIG_JSON_ENV}: {exc}") from exc
        return validate_model_config_payload(parsed, source=MODEL_CONFIG_JSON_ENV)

    env_config = os.environ.get("WORKFLOW_COMPILER_CONFIG")
    if env_config:
        return _load_model_config_from_path(env_config)

    return _load_model_config_from_path(get_config_path())


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = get_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


@lru_cache(maxsize=None)
def _load_model_alias_to_hf_name_map_cached(payload_key: str) -> Dict[str, str]:
    payload = validate_model_config_payload(json.loads(payload_key), source="cached_model_config")
    models = payload.get("models")
    alias_to_hf_name: Dict[str, str] = {}
    for cfg_key, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            continue

        hf_name = model_cfg.get("hf_model_name")
        if not hf_name:
            continue

        aliases = {str(cfg_key).strip()}
        explicit_alias = model_cfg.get("model")
        if explicit_alias:
            aliases.add(str(explicit_alias).strip())

        for alias in aliases:
            if not alias:
                continue
            existing = alias_to_hf_name.get(alias)
            if existing is not None and existing != str(hf_name):
                raise ValueError(
                    f"Conflicting hf_model_name values for alias '{alias}' in model config: "
                    f"{existing!r} vs {hf_name!r}"
                )
            alias_to_hf_name[alias] = str(hf_name)

    return alias_to_hf_name


def load_model_alias_to_hf_name_map(config_path: Optional[Union[str, Path, Dict[str, Any]]] = None) -> Dict[str, str]:
    """Load alias -> HuggingFace model mapping from the model YAML."""
    payload = load_model_config_payload(config_path)
    payload_key = json.dumps(payload, sort_keys=True)
    return dict(_load_model_alias_to_hf_name_map_cached(payload_key))


def parse_llm_config(config_str: str) -> Dict[str, Any]:
    """Parse config string via model-name extraction helpers."""
    from workflow_compiler.core.analysis.modeling import extract_model_name

    model, budget = extract_model_name(config_str, return_budget=True)
    return {"model": model, "budget": budget}


def create_experiment_config(
    name: str,
    benchmark: str,
    workflow_type: str,
    output_dir: str,
    **kwargs,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        benchmark=benchmark,
        workflow_type=workflow_type,
        output_dir=output_dir,
        llm_configs=kwargs.get("llm_configs", {}),
        router_config=kwargs.get("router_config"),
        eval_config=kwargs.get("eval_config", {}),
    )


def validate_config(config: Dict[str, Any]) -> bool:
    if "llm" not in config and "models" not in config:
        raise ValueError("Config must contain 'llm' or 'models' section")
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = [
    "ThinkingBudgetLLM",
    "parse_config",
    "build_setting",
    "LLMConfig",
    "ExperimentConfig",
    "get_config_path",
    "load_config",
    "load_model_alias_to_hf_name_map",
    "parse_llm_config",
    "create_experiment_config",
    "validate_config",
    "merge_configs",
]
