"""In-process vLLM backend for FlashFlow."""
from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from flashflow.backends.base import BaseBackend
from flashflow.thinking import (
    DEFAULT_THINKING_BUDGET_CUTOFF_TEXT,
    FlashFlowThinkingBudgetLogitsProcessor,
    THINK_END_TEXT,
    THINKING_BUDGET_EXTRA_ARG,
    THINKING_END_TOKEN_IDS_ARG,
    THINKING_SUFFIX_TOKEN_IDS_ARG,
)
from flashflow.types import AliasInfo, BackendResult, extract_request_common


def _normalize_stop(stop: Any) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, (list, tuple)):
        return [str(item) for item in stop]
    return [str(stop)]


class VLLMBackend(BaseBackend):
    def __init__(self, model_name: str, metadata: Dict[str, Any], vllm_args: Dict[str, Any]) -> None:
        super().__init__(model_name, metadata)
        self.vllm_args = dict(vllm_args)
        self._llm = None
        self._sampling_params_cls = None
        self._tokenizer = None
        self._initialized = False

    @property
    def artifact_id(self) -> str:
        return str(self.metadata.get("artifact_id") or self.model_name)

    @property
    def tokenizer_name(self) -> str:
        return str(self.metadata.get("tokenizer") or self.artifact_id)

    async def initialize(self) -> None:
        if self._initialized:
            return
        await asyncio.to_thread(self._prefetch_model_artifacts)
        await asyncio.to_thread(self._initialize_sync)
        self._initialized = True

    def _initialize_sync(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        kwargs = dict(self.vllm_args)
        kwargs.setdefault("model", self.artifact_id)
        kwargs.setdefault("tokenizer", self.tokenizer_name)
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("enable_sleep_mode", True)
        if "logits_processors" not in inspect.signature(LLM).parameters:
            raise ValueError(
                "The installed vLLM build does not expose engine-level logits_processors on LLM(...)."
            )
        existing_logits_processors = list(kwargs.pop("logits_processors", []) or [])
        existing_logits_processors.append(FlashFlowThinkingBudgetLogitsProcessor)
        kwargs["logits_processors"] = existing_logits_processors
        self._llm = LLM(**kwargs)
        self._sampling_params_cls = SamplingParams
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True,
        )

    def _prefetch_model_artifacts(self) -> None:
        artifact = self.artifact_id
        paths: List[Path] = []
        artifact_path = Path(artifact)
        if artifact_path.exists():
            if artifact_path.is_dir():
                paths.extend(sorted(artifact_path.rglob("*")))
            else:
                paths.append(artifact_path)
        else:
            try:
                from huggingface_hub import snapshot_download

                local_path = Path(
                    snapshot_download(
                        repo_id=artifact,
                        local_files_only=False,
                    )
                )
                paths.extend(sorted(local_path.rglob("*")))
            except Exception:
                return
        for file_path in paths:
            if not file_path.is_file():
                continue
            if file_path.suffix not in {".json", ".model", ".txt", ".safetensors"}:
                continue
            try:
                with open(file_path, "rb") as handle:
                    while handle.read(1024 * 1024):
                        pass
            except Exception:
                continue

    async def warmup(self) -> None:
        if self._llm is None:
            return
        await asyncio.to_thread(
            self._generate_text,
            "Warm up the model.",
            self._make_sampling_params(max_tokens=1),
        )

    async def wake(self) -> None:
        if self._llm is None:
            return
        wake = getattr(self._llm, "wake_up", None)
        if wake:
            await asyncio.to_thread(wake)

    async def sleep(self, level: int = 1) -> None:
        if self._llm is None:
            return
        sleep = getattr(self._llm, "sleep", None)
        if sleep:
            await asyncio.to_thread(sleep, level, "wait")

    def _apply_chat_template(self, messages: List[Dict[str, Any]], enable_thinking: Optional[bool]) -> str:
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _make_sampling_params(self, **overrides):
        params = {
            "temperature": overrides.pop("temperature", 1),
            "top_p": overrides.pop("top_p", 1),
            "max_tokens": overrides.pop("max_tokens", 512),
        }
        stop = overrides.pop("stop", None)
        if stop:
            params["stop"] = stop
        params.update(overrides)
        return self._sampling_params_cls(**params)

    def _encode(self, text: str) -> Sequence[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _generate_text(self, prompt: str, sampling_params) -> Tuple[str, Sequence[int]]:
        outputs = self._llm.generate([prompt], sampling_params, use_tqdm=False)
        result = outputs[0].outputs[0]
        return result.text, getattr(result, "token_ids", [])

    async def _generate(self, prompt: str, sampling_params) -> Tuple[str, Sequence[int]]:
        return await asyncio.to_thread(self._generate_text, prompt, sampling_params)

    async def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        alias_info: AliasInfo,
        request: Dict[str, Any],
    ) -> BackendResult:
        prompt = self._build_chat_prompt(messages, alias_info)
        return await self._generate_from_prompt(prompt, alias_info, request)

    async def generate_completion(
        self,
        prompt: str,
        alias_info: AliasInfo,
        request: Dict[str, Any],
    ) -> BackendResult:
        return await self._generate_from_prompt(prompt, alias_info, request)

    def _build_chat_prompt(self, messages: List[Dict[str, Any]], alias_info: AliasInfo) -> str:
        enable_thinking = None
        if alias_info.budget == "nothinking":
            enable_thinking = False
        elif alias_info.budget not in (None, "unlimited"):
            enable_thinking = True
        return self._apply_chat_template(messages, enable_thinking=enable_thinking)

    async def _generate_from_prompt(
        self,
        prompt: str,
        alias_info: AliasInfo,
        request: Dict[str, Any],
    ) -> BackendResult:
        options = extract_request_common(request)
        prompt_tokens = len(self._encode(prompt))
        budget = alias_info.budget
        strategy = alias_info.default_thinking_strategy
        if budget in (None, "unlimited"):
            text, output_ids = await self._generate(
                prompt,
                self._make_sampling_params(
                    temperature=options["temperature"],
                    top_p=options["top_p"],
                    max_tokens=options["max_tokens"] or 512,
                    stop=_normalize_stop(options["stop"]),
                ),
            )
            return BackendResult(
                text=text.split(THINK_END_TEXT)[-1].strip() if text else "",
                input_tokens=prompt_tokens,
                output_tokens=len(output_ids),
                model_name=self.model_name,
            )
        if budget == "nothinking":
            text, output_ids = await self._generate(
                prompt,
                self._make_sampling_params(
                    temperature=options["temperature"],
                    top_p=options["top_p"],
                    max_tokens=options["max_tokens"] or 512,
                    stop=_normalize_stop(options["stop"]),
                ),
            )
            return BackendResult(
                text=text.split(THINK_END_TEXT)[-1].strip() if text else "",
                input_tokens=prompt_tokens,
                output_tokens=len(output_ids),
                model_name=self.model_name,
            )
        if not isinstance(budget, int):
            raise ValueError(f"Unsupported thinking budget '{budget}' for model '{self.model_name}'.")
        if strategy == "two_stage":
            raise ValueError(
                f"FlashFlow vLLM backend requires plugin-based thinking budget for '{self.model_name}'; "
                "two-stage fallback is disabled."
            )
        return await self._generate_with_plugin(prompt, budget, options)

    async def _generate_with_plugin(
        self,
        prompt: str,
        budget: int,
        options: Dict[str, Any],
    ) -> BackendResult:
        prompt_tokens = len(self._encode(prompt))
        text, output_ids = await self._generate(
            prompt,
            self._make_sampling_params(
                temperature=options["temperature"],
                top_p=options["top_p"],
                max_tokens=options["max_tokens"] or 512,
                stop=_normalize_stop(options["stop"]),
                extra_args={
                    THINKING_BUDGET_EXTRA_ARG: int(budget),
                    THINKING_SUFFIX_TOKEN_IDS_ARG: list(
                        self._encode(DEFAULT_THINKING_BUDGET_CUTOFF_TEXT)
                    ),
                    THINKING_END_TOKEN_IDS_ARG: list(self._encode(THINK_END_TEXT)),
                },
            ),
        )
        return BackendResult(
            text=text.split(THINK_END_TEXT)[-1].strip() if text else "",
            input_tokens=prompt_tokens,
            output_tokens=len(output_ids),
            model_name=self.model_name,
        )

    async def _generate_two_stage(
        self,
        prompt: str,
        budget: int,
        options: Dict[str, Any],
    ) -> BackendResult:
        stage1_prompt_tokens = len(self._encode(prompt))
        stage1_text, stage1_output_ids = await self._generate(
            prompt,
            self._make_sampling_params(
                temperature=options["temperature"],
                top_p=options["top_p"],
                max_tokens=int(budget),
                stop=[THINK_END_TEXT],
            ),
        )
        if len(stage1_output_ids) >= int(budget):
            thinking_text = stage1_text + DEFAULT_THINKING_BUDGET_CUTOFF_TEXT
        else:
            thinking_text = stage1_text + THINK_END_TEXT + "\n\n"
        prompt2 = prompt + thinking_text
        stage2_text, stage2_output_ids = await self._generate(
            prompt2,
            self._make_sampling_params(
                temperature=options["temperature"],
                top_p=options["top_p"],
                max_tokens=options["max_tokens"] or 512,
                stop=_normalize_stop(options["stop"]),
            ),
        )
        stage2_prompt_tokens = len(self._encode(prompt2))
        synthetic_output_tokens = (stage2_prompt_tokens - stage1_prompt_tokens) + len(stage2_output_ids)
        return BackendResult(
            text=stage2_text,
            input_tokens=stage1_prompt_tokens,
            output_tokens=synthetic_output_tokens,
            model_name=self.model_name,
        )
