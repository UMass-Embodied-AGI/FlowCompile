# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang
# @Desc    : 

try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:  # pragma: no cover
    from openai import AsyncOpenAI
    AsyncAzureOpenAI = None
from workflow_compiler.core.llm.formatter import BaseFormatter, FormatError
from httpx import Timeout

import asyncio
import yaml
import os
from pathlib import Path
from typing import Dict, Optional, Any
from workflow_compiler.core.llm.thinking_budget import (
    DEFAULT_THINKING_BUDGET_CUTOFF_TEXT,
    DEFAULT_THINKING_BUDGET_REASONING_PARSER,
    DEFAULT_THINKING_BUDGET_VLLM_ARG_NAME,
    THINKING_BUDGET_ARG_NAME_ARG_NAME,
    THINKING_BUDGET_HF_MODEL_ARG_NAME,
    THINKING_CUTOFF_TEXT_ARG_NAME,
)
from workflow_compiler.core.llm.config import load_model_config_payload

class LLMConfig:
    def __init__(self, config: dict):
        self.name = config.get("name")
        self.api_type = str(config.get("api_type") or "openai").lower()
        self.model = (
            config.get("model")
            or config.get("azure_deployment")
            or config.get("deployment_name")
            or config.get("deployment")
            or "gpt-5-mini"
        )
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key") or config.get("api_key")
        self.base_url = config.get("base_url", "https://oneapi.deepwisdom.ai/v1")
        self.top_p = config.get("top_p", 1)
        self.api_version = config.get("api_version")
        self.azure_endpoint = (
            config.get("azure_endpoint")
            or config.get("endpoint")
            or self.base_url
        )
        self.azure_deployment = (
            config.get("azure_deployment")
            or config.get("deployment_name")
            or config.get("deployment")
            or self.model
        )
        # Retain raw config for downstream validation
        self.raw = config

class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        payload = config_dict or {}
        self.endpoints: Dict[str, Any] = {}
        if "models" in payload and isinstance(payload.get("models"), dict):
            self.configs = payload.get("models") or {}
            if isinstance(payload.get("endpoints"), dict):
                self.endpoints = dict(payload.get("endpoints") or {})
        else:
            self.configs = payload
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            config_data = load_model_config_payload()
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def _resolve_base_url(self, config: Dict[str, Any], endpoint_role: Optional[str] = None) -> Optional[str]:
        role = str(endpoint_role).strip().lower() if endpoint_role is not None else None
        if role not in {None, "latency", "profile"}:
            raise ValueError(f"Unsupported endpoint_role '{endpoint_role}'")

        if role == "latency":
            return self.endpoints.get("local_base_url") or config.get("base_url")
        if role == "profile":
            return self.endpoints.get("profile_base_url") or config.get("base_url")
        return config.get("base_url") or self.endpoints.get("local_base_url")

    def get(self, llm_name: str, endpoint_role: Optional[str] = None) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        
        # Create a config dictionary with the expected keys for LLMConfig
        llm_config = {
            **config,
            "name": llm_name,
            "model": config.get("model", llm_name),
            "temperature": config.get("temperature", 1),
            "top_p": config.get("top_p", 1),
        }

        if "key" not in llm_config:
            llm_config["key"] = llm_config.get("api_key")
        resolved_base_url = self._resolve_base_url(config, endpoint_role=endpoint_role)
        if resolved_base_url:
            llm_config["base_url"] = resolved_base_url
        elif "base_url" not in llm_config:
            llm_config["base_url"] = "https://oneapi.deepwisdom.ai/v1"
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class TokenUsageTracker:
    """Tracks token usage."""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }

    async def aclose(self):
        """No-op placeholder for compatibility with AsyncLLM cleanup"""
        return None

class AsyncLLM:
    def __init__(self, config, system_msg:str = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        
        # At this point, config should be an LLMConfig instance
        self.config = config
        assert self.config.temperature == 1
        assert self.config.top_p == 1
        
        # Initialize attributes that may be used across all API types
        self.enable_thinking_budget = False
        self.default_reasoning_effort = None
        self.thinking_budget_cutoff_text = DEFAULT_THINKING_BUDGET_CUTOFF_TEXT
        self.thinking_budget_reasoning_parser = (
            DEFAULT_THINKING_BUDGET_REASONING_PARSER
        )
        self.thinking_budget_vllm_arg_name = DEFAULT_THINKING_BUDGET_VLLM_ARG_NAME
        if self.config.api_type == "azure":
            if AsyncAzureOpenAI is None:
                raise ImportError("AsyncAzureOpenAI is unavailable. Please upgrade the openai package.")

            if not self.config.key:
                raise ValueError("Azure configuration requires an api_key.")
            if not self.config.azure_endpoint:
                raise ValueError("Azure configuration requires an azure_endpoint.")
            if not self.config.azure_deployment:
                raise ValueError("Azure configuration requires an azure_deployment or model name.")

            api_version = self.config.api_version or "2024-02-15-preview"
            # Set a longer timeout: 30 minutes (1800 seconds) for connect, read, write, and pool operations
            timeout = Timeout(timeout=1800.0, connect=60.0)
            self.aclient = AsyncAzureOpenAI(
                azure_endpoint=self.config.azure_endpoint,
                api_key=self.config.key,
                api_version=api_version,
                timeout=timeout,
            )
            self._request_model = self.config.azure_deployment
        elif self.config.api_type == "openai":
            # Set a longer timeout: 30 minutes (1800 seconds) for connect, read, write, and pool operations
            timeout = Timeout(timeout=1800.0, connect=60.0)
            self.aclient = AsyncOpenAI(
                api_key=self.config.key, 
                base_url=self.config.base_url,
                timeout=timeout
            )
            self._request_model = self.config.model
            self.enable_thinking_budget = self.config.raw["enable_thinking_budget"] if hasattr(self.config, 'raw') and "enable_thinking_budget" in self.config.raw else False
            self.default_reasoning_effort = self.config.raw.get("default_reasoning_effort") if hasattr(self.config, 'raw') else None
            self.thinking_budget_cutoff_text = (
                self.config.raw.get("thinking_budget_cutoff_text")
                if hasattr(self.config, "raw")
                else None
            ) or DEFAULT_THINKING_BUDGET_CUTOFF_TEXT
            self.thinking_budget_reasoning_parser = (
                self.config.raw.get("thinking_budget_reasoning_parser")
                if hasattr(self.config, "raw")
                else None
            ) or DEFAULT_THINKING_BUDGET_REASONING_PARSER
            self.thinking_budget_vllm_arg_name = (
                self.config.raw.get("thinking_budget_vllm_arg_name")
                if hasattr(self.config, "raw")
                else None
            ) or DEFAULT_THINKING_BUDGET_VLLM_ARG_NAME
        else:
            raise ValueError(
                f"Unsupported api_type '{self.config.api_type}'. Expected 'openai' or 'azure'."
            )
        assert system_msg is None, "System message support is deprecated."
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()

    async def __call__(self, prompt, return_io_tokens: bool = False, disable_thinking: bool = False):
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        message.append({"role": "user", "content": prompt})

        # Prepare extra_body for disabling thinking if needed
        extra_body = {}
        if disable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        elif self.default_reasoning_effort:
            # Use default reasoning effort if configured
            extra_body = {"reasoning_effort": self.default_reasoning_effort}

        response = await self.aclient.chat.completions.create(
            model=self._request_model,
            messages=message,
            temperature=self.config.temperature,
            top_p = self.config.top_p,
            **({"extra_body": extra_body} if extra_body else {})
        )

        # Extract token usage from response when available
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        
        # Track token usage
        self.usage_tracker.add_usage(
            self._request_model,
            input_tokens,
            output_tokens
        )
        content = response.choices[0].message.content
        ret = content.split("</think>")[-1].strip() if content else ""
        
        if return_io_tokens:
            return ret, input_tokens, output_tokens
        return ret
    
    async def call_with_format(self, prompt: str, formatter: BaseFormatter, return_io_tokens: bool = False):
        """
        Call the LLM with a prompt and format the response using the provided formatter
        
        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response
            return_io_tokens: If True, return (parsed_data, input_tokens, output_tokens)
            
        Returns:
            The formatted response data, or tuple if return_io_tokens=True
            
        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM with token tracking
        result = await self.__call__(formatted_prompt, return_io_tokens=True)
        response, input_tokens, output_tokens = result
        
        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)
        
        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}.")
        
        if return_io_tokens:
            return parsed_data, input_tokens, output_tokens
        return parsed_data
    
    async def call_with_tools(self, messages: list, tools: list, tool_choice: str = "auto", return_io_tokens: bool = False):
        """
        Call the LLM with OpenAI-compatible tool calling
        
        Args:
            messages: List of message dicts with role and content
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or specific tool name
            return_io_tokens: If True, return (message, input_tokens, output_tokens)
            
        Returns:
            The response message object with potential tool_calls, or tuple if return_io_tokens=True
        """
        response = await self.aclient.chat.completions.create(
            model=self._request_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        # Track token usage
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        
        self.usage_tracker.add_usage(
            self._request_model,
            input_tokens,
            output_tokens
        )
        content = response.choices[0].message.content
        response.choices[0].message.content = content.split("</think>")[-1].strip() if content else ""
        
        if return_io_tokens:
            return response.choices[0].message, input_tokens, output_tokens
        return response.choices[0].message
    
    async def call_with_thinking_budget(self, prompt: str, thinking_budget: Optional[int | str], return_io_tokens: bool = False):
        if isinstance(thinking_budget, str):
            message = []
            if self.sys_msg is not None:
                message.append({
                    "content": self.sys_msg,
                    "role": "system"
                })
            message.append({"role": "user", "content": prompt})
            if thinking_budget == "unlimited":
                response = await self.aclient.chat.completions.create(
                    model=self._request_model,
                    messages=message,
                    temperature=self.config.temperature,
                    top_p = self.config.top_p,
                )
                content = response.choices[0].message.content
                response.choices[0].message.content = content.split("</think>")[-1].strip() if content else ""
            elif thinking_budget == "nothinking":
                # Disable thinking mode
                response = await self.aclient.chat.completions.create(
                    model=self._request_model,
                    messages=message,
                    temperature=self.config.temperature,
                    top_p = self.config.top_p,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
            else:
                # GPT-OSS style budget string
                response = await self.aclient.chat.completions.create(
                    model=self._request_model,
                    messages=message,
                    temperature=self.config.temperature,
                    top_p = self.config.top_p,
                    extra_body={"reasoning_effort": thinking_budget}
                )
            # Extract token usage from response when available
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            # Track token usage
            self.usage_tracker.add_usage(
                self._request_model,
                input_tokens,
                output_tokens
            )
            ret = response.choices[0].message.content
            if return_io_tokens:
                return ret, input_tokens, output_tokens
            else:
                return ret

        if not self.enable_thinking_budget:
            raise ValueError("Thinking budget feature is not enabled for this model configuration.")
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })
        message.append({"role": "user", "content": prompt})
        vllm_xargs = {
            self.thinking_budget_vllm_arg_name: int(thinking_budget),
            THINKING_CUTOFF_TEXT_ARG_NAME: self.thinking_budget_cutoff_text,
            THINKING_BUDGET_ARG_NAME_ARG_NAME: self.thinking_budget_vllm_arg_name,
        }
        hf_model_name = None
        if hasattr(self.config, "raw"):
            hf_model_name = self.config.raw.get("hf_model_name")
        if hf_model_name:
            vllm_xargs[THINKING_BUDGET_HF_MODEL_ARG_NAME] = hf_model_name
        try:
            response = await self.aclient.chat.completions.create(
                model=self._request_model,
                messages=message,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "vllm_xargs": vllm_xargs,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                "Integer thinking budgets via vLLM logits processor failed. "
                "Validate that LiteLLM preserves extra_body.vllm_xargs."
            ) from exc

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.usage_tracker.add_usage(
            self._request_model,
            input_tokens,
            output_tokens
        )
        content = response.choices[0].message.content
        final_response = content.split("</think>")[-1].strip() if content else ""
        if return_io_tokens:
            return final_response, input_tokens, output_tokens
        return final_response


    def get_usage_summary(self):
        """Get a summary of token usage."""
        return self.usage_tracker.get_summary()    

    async def aclose(self):
        """Close underlying async client to release HTTP sessions/sockets"""
        close_method = getattr(self.aclient, "aclose", None) or getattr(self.aclient, "close", None)
        if close_method:
            try:
                maybe_coro = close_method()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
            except Exception:
                pass
        await self.usage_tracker.aclose()
    

def create_llm_instance(llm_config, endpoint_role: Optional[str] = None):
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
        endpoint_role: Optional route role. Supported values: "latency", "profile".
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        if endpoint_role:
            try:
                refreshed = LLMsConfig.default().get(
                    llm_config.name or llm_config.model,
                    endpoint_role=endpoint_role,
                )
                return AsyncLLM(refreshed)
            except ValueError:
                pass
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        resolved = LLMsConfig.default().get(llm_config, endpoint_role=endpoint_role)
        return AsyncLLM(resolved)
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
