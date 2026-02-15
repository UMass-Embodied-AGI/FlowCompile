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
from urllib.parse import urlparse
from transformers import AutoTokenizer

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
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            # Look for the config file in common locations
            env_config = os.environ.get("WORKFLOW_COMPILER_CONFIG")
            if env_config and Path(env_config).exists():
                config_file = Path(env_config)
            else:
                config_file = None
            config_paths = [
                Path("configs/config.yaml"),
                Path("config.yaml"),
                Path("./configs/config.yaml"),
                # Backward-compatible fallbacks
                Path("configs/config2.yaml"),
                Path("config2.yaml"),
                Path("./configs/config2.yaml"),
            ]
            if config_file is None:
                for path in config_paths:
                    if path.exists():
                        config_file = path
                        break
            
            if config_file is None:
                raise FileNotFoundError("No default configuration file found in the expected locations")
            
            # Load the YAML file
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Your YAML has a 'models' top-level key that contains the model configs
            if 'models' in config_data:
                config_data = config_data['models']
                
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def get(self, llm_name: str) -> LLMConfig:
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
        if "base_url" not in llm_config:
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
    _TOKENIZER_CACHE: Dict[tuple, Any] = {}

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
            if self.enable_thinking_budget:
                self.tokenizer = self._load_tokenizer()
        else:
            raise ValueError(
                f"Unsupported api_type '{self.config.api_type}'. Expected 'openai' or 'azure'."
            )
        assert system_msg is None, "System message support is deprecated."
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()

    def _is_local_backend(self) -> bool:
        if self.config.api_type != "openai":
            return False
        base_url = str(getattr(self.config, "base_url", "") or "")
        if not base_url:
            return False
        parsed = urlparse(base_url)
        hostname = (parsed.hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    def _load_tokenizer(self):
        hf_model_name = None
        if hasattr(self.config, "raw"):
            hf_model_name = self.config.raw.get("hf_model_name")
        if not hf_model_name:
            raise ValueError(
                "enable_thinking_budget=true requires 'hf_model_name' in model config."
            )

        force_local_only_env = os.environ.get("FLOWCOMPILE_HF_LOCAL_FILES_ONLY", "").strip().lower()
        if force_local_only_env in ("1", "true", "yes"):
            local_files_only = True
        elif force_local_only_env in ("0", "false", "no"):
            local_files_only = False
        else:
            # Default to local-only for localhost backends to avoid HF Hub rate-limit traffic.
            local_files_only = self._is_local_backend()

        cache_key = (str(hf_model_name), bool(local_files_only))
        cached = self._TOKENIZER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        kwargs = {"trust_remote_code": True}
        if local_files_only:
            kwargs["local_files_only"] = True

        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, **kwargs)
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    f"Failed to load local tokenizer for '{hf_model_name}' with local_files_only=True. "
                    "Pre-download tokenizer files to HF cache or set FLOWCOMPILE_HF_LOCAL_FILES_ONLY=0 "
                    "to allow Hub access."
                ) from exc
            raise

        self._TOKENIZER_CACHE[cache_key] = tokenizer
        return tokenizer

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

        early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
        if not self.enable_thinking_budget:
            raise ValueError("Thinking budget feature is not enabled for this model configuration.")
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })
        message.append({"role": "user", "content": prompt})
        prompt = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=False,
        )
        response1 = await self.aclient.completions.create(
            model=self._request_model,
            prompt=prompt,
            max_tokens=thinking_budget,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=["</think>"],
        )
        input_tokens = getattr(response1.usage, "prompt_tokens", 0) or 0
        if response1.choices[0].finish_reason == "length":
            thinking_text = response1.choices[0].text + early_stopping_text
        else:
            thinking_text = response1.choices[0].text + "</think>\n\n"
        prompt2 = prompt + thinking_text
        response2 = await self.aclient.completions.create(
            model=self._request_model,
            prompt=prompt2,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=16*1024,
        )
        final_response = response2.choices[0].text
        # Track token usage for the last call
        usage = getattr(response2, "usage", None)
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        second_input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = second_input_tokens - input_tokens + output_tokens
        self.usage_tracker.add_usage(
            self._request_model,
            input_tokens,
            output_tokens
        )
        if return_io_tokens:
            return final_response, input_tokens, output_tokens
        else:
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
    

def create_llm_instance(llm_config):
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
