from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer

from workflow_compiler.core.llm.thinking_budget import (
    DEFAULT_THINKING_BUDGET_CUTOFF_TEXT,
    DEFAULT_THINKING_BUDGET_VLLM_ARG_NAME,
    THINK_END_TEXT,
    THINKING_BUDGET_ARG_NAME_ARG_NAME,
    THINKING_BUDGET_HF_MODEL_ARG_NAME,
    THINKING_CUTOFF_TEXT_ARG_NAME,
    ThinkingBudgetState,
)

try:  # pragma: no cover - exercised in vLLM runtime, not local unit tests.
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import (
        AdapterLogitsProcessor,
        RequestLogitsProcessor,
    )
except ImportError:  # pragma: no cover
    SamplingParams = object
    RequestLogitsProcessor = object

    class AdapterLogitsProcessor:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass


class _ThinkingBudgetRequestProcessor:
    def __init__(self, state: ThinkingBudgetState):
        self.state = state

    def __call__(self, output_ids, logits):
        return self.state.apply(output_ids, logits)


class ThinkingBudgetLogitsProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        self._tokenizer_name = self._resolve_tokenizer_name(vllm_config)
        self._tokenizer = None
        self._encoded_cache: dict[str, tuple[int, ...]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        extra_args = getattr(params, "extra_args", None) or {}
        arg_name = str(
            extra_args.get(THINKING_BUDGET_ARG_NAME_ARG_NAME)
            or DEFAULT_THINKING_BUDGET_VLLM_ARG_NAME
        )
        raw_budget = extra_args.get(arg_name)
        if raw_budget is None:
            raw_budget = extra_args.get(DEFAULT_THINKING_BUDGET_VLLM_ARG_NAME)
        try:
            budget = int(raw_budget)
        except (TypeError, ValueError):
            return None
        if budget < 0:
            return None

        cutoff_text = str(
            extra_args.get(THINKING_CUTOFF_TEXT_ARG_NAME)
            or DEFAULT_THINKING_BUDGET_CUTOFF_TEXT
        )
        self._ensure_tokenizer(extra_args)
        state = ThinkingBudgetState(
            budget=budget,
            suffix_token_ids=self._encode_text(cutoff_text),
            end_think_token_ids=self._encode_text(THINK_END_TEXT),
        )
        return _ThinkingBudgetRequestProcessor(state)

    def _ensure_tokenizer(self, extra_args) -> None:
        if self._tokenizer is not None:
            return
        tokenizer_name = (
            self._tokenizer_name
            or extra_args.get(THINKING_BUDGET_HF_MODEL_ARG_NAME)
        )
        if not tokenizer_name:
            raise ValueError(
                "ThinkingBudgetLogitsProcessor could not determine tokenizer name."
            )
        self._tokenizer_name = str(tokenizer_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_name,
            trust_remote_code=True,
        )

    def _encode_text(self, text: str) -> tuple[int, ...]:
        cached = self._encoded_cache.get(text)
        if cached is not None:
            return cached
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before encoding text.")
        token_ids = tuple(
            self._tokenizer.encode(text, add_special_tokens=False)
        )
        self._encoded_cache[text] = token_ids
        return token_ids

    @staticmethod
    def _resolve_tokenizer_name(vllm_config) -> Optional[str]:
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            return None
        tokenizer_name = getattr(model_config, "tokenizer", None)
        if tokenizer_name:
            return str(tokenizer_name)
        model_name = getattr(model_config, "model", None)
        if model_name:
            return str(model_name)
        return None
