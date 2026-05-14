"""Latency benchmarking utilities for FlowCompile."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from flowcompile.core.terminal import get_reporter


def _configure_cuda_multiprocessing() -> None:
    """Force spawn-based multiprocessing for CUDA-safe worker startup."""
    # vLLM respects this environment variable for worker process creation.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    try:
        import multiprocessing as mp
        current = mp.get_start_method(allow_none=True)
        if current != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Context may already be initialized by parent process.
                pass
    except Exception:
        pass

    try:
        import torch.multiprocessing as tmp
        current = tmp.get_start_method(allow_none=True)
        if current != "spawn":
            try:
                tmp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Context may already be initialized by parent process.
                pass
    except Exception:
        pass


def _require_latency_deps():
    try:
        from vllm import SamplingParams, AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Latency benchmarking requires `vllm` and `transformers` to be installed."
        ) from exc
    return SamplingParams, AsyncLLMEngine, AsyncEngineArgs, AutoTokenizer


def _gpu_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _require_openai_dep():
    try:
        from openai import AsyncOpenAI
    except Exception as exc:
        raise RuntimeError(
            "OpenAI-compatible latency benchmarking requires `openai` to be installed."
        ) from exc
    return AsyncOpenAI


def _require_yaml_dep():
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "OpenAI-compatible latency benchmarking requires `pyyaml` to be installed."
        ) from exc
    return yaml

@dataclass
class BatchStats:
    batch_size: int
    total_prompt_tokens: int
    total_generated_tokens: int
    ttft_avg_s: float
    ttft_p95_s: float
    prefill_time_s: float        # we use max TTFT across the batch
    decode_time_s: float         # from earliest first token to last finish
    prefill_tok_per_s: Optional[float]
    decode_tok_per_s: Optional[float]

async def _stream_one(
    engine,
    req_id: str,
    prompt: str,
    sampling_params,
) -> Tuple[float, float, int]:
    """
    Stream a single request; return (t_start, t_first, gen_tokens, t_end) as:
    - t_start is captured by caller before submitting the request
    - t_first: time at first generated token
    - gen_tokens: final generated token count
    - t_end: time when generation finishes
    """
    first_time = None
    last_len = 0

    # Iterate streamed RequestOutput objects
    async for out in engine.generate(
        request_id=req_id,
        prompt=prompt,
        sampling_params=sampling_params,
    ):
        # out.outputs[0].token_ids grows as we stream
        if out.outputs:
            cur_len = len(out.outputs[0].token_ids)
            if first_time is None and cur_len > 0:
                _gpu_sync()
                first_time = time.perf_counter()
            last_len = max(last_len, cur_len)

    _gpu_sync()
    end_time = time.perf_counter()
    if first_time is None:
        first_time = end_time  # nothing decoded (e.g., max_new_tokens=0)

    return first_time, end_time, last_len

async def measure_batch(
    engine,
    tokenizer,
    prompt_text: str,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
) -> BatchStats:
    # Tokenizer for counting prompt tokens
    prompt_tok_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    total_prompt_tokens = prompt_tok_len * batch_size

    # Real run
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        # return_token_ids is handled by Python API automatically
    )

    prompts = [prompt_text] * batch_size
    req_ids = [f"req-{i}" for i in range(batch_size)]

    _gpu_sync()
    t0 = time.perf_counter()

    coros = [
        _stream_one(engine, rid, p, sp)
        for rid, p in zip(req_ids, prompts)
    ]

    results: List[Tuple[float, float, int]] = []
    try:
        async for r in _async_drain(coros):
            results.append(r)
    finally:
        pass  # No shutdown here, done outside

    # Extract per-request timings
    first_times = [r[0] for r in results]
    end_times   = [r[1] for r in results]
    gen_tokens  = [r[2] for r in results]

    ttfts = [ft - t0 for ft in first_times]
    ttft_avg = sum(ttfts) / len(ttfts) if ttfts else 0.0
    ttft_p95 = sorted(ttfts)[int(0.95 * (len(ttfts) - 1))] if ttfts else 0.0

    # Batch-level phase windows
    prefill_time = max(ttfts) if ttfts else 0.0                  # slowest TTFT
    decode_time  = (max(end_times) - min(first_times)) if results else 0.0

    total_generated = sum(gen_tokens)

    prefill_tps = (total_prompt_tokens / prefill_time) if prefill_time > 0 else None
    decode_tps  = (total_generated / decode_time) if decode_time > 0 else None

    return BatchStats(
        batch_size=batch_size,
        total_prompt_tokens=total_prompt_tokens,
        total_generated_tokens=total_generated,
        ttft_avg_s=ttft_avg,
        ttft_p95_s=ttft_p95,
        prefill_time_s=prefill_time,
        decode_time_s=decode_time,
        prefill_tok_per_s=prefill_tps,
        decode_tok_per_s=decode_tps,
    )

async def _async_drain(coros):
    # Run many coroutines and yield results as they complete
    tasks = {asyncio.create_task(c): c for c in coros}
    while tasks:
        done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            tasks.pop(t, None)
            yield t.result()

def format_float(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "n/a"

def _normalize_models(models: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(models, str):
        return [m.strip() for m in models.split(",") if m.strip()]
    return [str(m).strip() for m in models if str(m).strip()]


def _normalize_batch_sizes(batch_sizes: Optional[Union[str, Sequence[int]]], batch_size: int) -> List[int]:
    if batch_sizes is None:
        return [batch_size]
    if isinstance(batch_sizes, str):
        parsed = [int(x) for x in batch_sizes.split(",") if x.strip()]
        return parsed or [batch_size]
    return [int(x) for x in batch_sizes] if batch_sizes else [batch_size]


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _load_model_routes(model_config_path: str) -> List[Dict[str, Any]]:
    yaml = _require_yaml_dep()

    with open(model_config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_models = data.get("models", data)
    if not isinstance(raw_models, dict):
        raise ValueError(f"Invalid model config format in {model_config_path}")

    routes: List[Dict[str, Any]] = []
    for name, cfg in raw_models.items():
        if not isinstance(cfg, dict):
            continue
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key") or cfg.get("key") or os.environ.get("OPENAI_API_KEY")
        if not base_url or not api_key:
            continue

        request_model = (
            cfg.get("model")
            or cfg.get("azure_deployment")
            or cfg.get("deployment_name")
            or cfg.get("deployment")
            or name
        )
        hf_model_name = cfg.get("hf_model_name")
        aliases = {str(name), str(request_model)}
        if hf_model_name:
            aliases.add(str(hf_model_name))

        routes.append(
            {
                "name": str(name),
                "request_model": str(request_model),
                "base_url": str(base_url),
                "api_key": str(api_key),
                "aliases": aliases,
            }
        )
    return routes


def _resolve_route(model: str, routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for route in routes:
        if model in route["aliases"]:
            return route

    model_lower = model.lower()
    for route in routes:
        if any(alias.lower() == model_lower for alias in route["aliases"]):
            return route
    return None


async def _create_stream_with_fallbacks(client, payload: Dict[str, Any]):
    attempts: List[Dict[str, Any]] = [dict(payload)]
    payload_no_seed = dict(payload)
    payload_no_seed.pop("seed", None)
    attempts.append(payload_no_seed)
    payload_min = dict(payload_no_seed)
    payload_min.pop("stream_options", None)
    attempts.append(payload_min)

    last_exc = None
    for request in attempts:
        try:
            return await client.chat.completions.create(**request)
        except Exception as exc:
            last_exc = exc
    raise last_exc


async def _stream_one_openai(
    client,
    request_model: str,
    prompt: str,
    max_new_tokens: int,
    seed: int,
) -> Tuple[float, float, Optional[int], Optional[int]]:
    first_time = None
    prompt_tokens = None
    completion_tokens = None
    generated_text: List[str] = []

    payload = {
        "model": request_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": seed,
    }
    stream = await _create_stream_with_fallbacks(client, payload)

    async for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            p = _maybe_int(getattr(usage, "prompt_tokens", None))
            c = _maybe_int(getattr(usage, "completion_tokens", None))
            if p is not None:
                prompt_tokens = p
            if c is not None:
                completion_tokens = c

        if not chunk.choices:
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if first_time is None and (content or reasoning):
            _gpu_sync()
            first_time = time.perf_counter()
        if content:
            generated_text.append(content)

    _gpu_sync()
    end_time = time.perf_counter()
    if first_time is None:
        first_time = end_time

    if completion_tokens is None:
        completion_tokens = _estimate_tokens("".join(generated_text))

    return first_time, end_time, prompt_tokens, completion_tokens


async def measure_batch_openai(
    client,
    request_model: str,
    prompt_text: str,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
) -> BatchStats:
    estimated_prompt_tokens = _estimate_tokens(prompt_text)
    coros = [
        _stream_one_openai(client, request_model, prompt_text, max_new_tokens, seed + i)
        for i in range(batch_size)
    ]

    _gpu_sync()
    t0 = time.perf_counter()

    results: List[Tuple[float, float, Optional[int], Optional[int]]] = []
    async for r in _async_drain(coros):
        results.append(r)

    first_times = [r[0] for r in results]
    end_times = [r[1] for r in results]
    prompt_tokens = [r[2] if r[2] is not None else estimated_prompt_tokens for r in results]
    completion_tokens = [r[3] if r[3] is not None else 0 for r in results]

    ttfts = [ft - t0 for ft in first_times]
    ttft_avg = sum(ttfts) / len(ttfts) if ttfts else 0.0
    ttft_p95 = sorted(ttfts)[int(0.95 * (len(ttfts) - 1))] if ttfts else 0.0

    prefill_time = max(ttfts) if ttfts else 0.0
    decode_time = (max(end_times) - min(first_times)) if results else 0.0

    total_prompt_tokens = sum(int(x) for x in prompt_tokens)
    total_generated = sum(int(x) for x in completion_tokens)
    prefill_tps = (total_prompt_tokens / prefill_time) if prefill_time > 0 else None
    decode_tps = (total_generated / decode_time) if decode_time > 0 else None

    return BatchStats(
        batch_size=batch_size,
        total_prompt_tokens=total_prompt_tokens,
        total_generated_tokens=total_generated,
        ttft_avg_s=ttft_avg,
        ttft_p95_s=ttft_p95,
        prefill_time_s=prefill_time,
        decode_time_s=decode_time,
        prefill_tok_per_s=prefill_tps,
        decode_tok_per_s=decode_tps,
    )


def _print_results_table(model: str, prompt_file: str, max_new_tokens: int, results: List[BatchStats]) -> None:
    reporter = get_reporter().child("get-latency")
    reporter.detail(f"Throughput profile for {model}")
    reporter.detail(f"Model: {model}")
    reporter.detail(f"Prompt file: {prompt_file}")
    reporter.detail(f"Max new tokens: {max_new_tokens}")
    hdr = (
        "Batch  | PromptToks | GenToks | TTFT_avg  TTFT_p95 | "
        "Prefill_s  Decode_s | Prefill tok/s  Decode tok/s"
    )
    reporter.detail(hdr)
    for r in results:
        reporter.detail(
            f"{r.batch_size:5d} |"
            f" {r.total_prompt_tokens:10d} |"
            f" {r.total_generated_tokens:7d} |"
            f" {r.ttft_avg_s:8.3f} {r.ttft_p95_s:8.3f} |"
            f" {r.prefill_time_s:8.3f} {r.decode_time_s:8.3f} |"
            f" {format_float(r.prefill_tok_per_s):>14} {format_float(r.decode_tok_per_s):>13}"
        )


def _save_latency_json(output_json: str, payload: Dict[str, Any]) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_latency_benchmark_vllm(
    models_list: List[str],
    prompt: str,
    prompt_file: str,
    output_json: str,
    batch_sizes_list: List[int],
    max_new_tokens: int,
    dtype: str,
    tp: int,
    gpu_mem_util: float,
    seed: int,
) -> Dict[str, Any]:
    SamplingParams, AsyncLLMEngine, AsyncEngineArgs, AutoTokenizer = _require_latency_deps()

    async def _run_all(model: str):
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        eng = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model,
                dtype=dtype,
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_mem_util,
                trust_remote_code=True,
                max_model_len=32 * 1024,
            )
        )

        warm_sp = SamplingParams(max_tokens=8, temperature=0.0, top_p=1.0, seed=seed)
        warmups = [_stream_one(eng, "warmup-0", prompt, warm_sp)]
        async for _ in _async_drain(warmups):
            pass

        results = []
        for bs in batch_sizes_list:
            res = await measure_batch(
                engine=eng,
                tokenizer=tok,
                prompt_text=prompt,
                batch_size=bs,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            results.append(res)

        try:
            eng.shutdown()
        except Exception:
            pass
        return results

    all_results: Dict[str, Any] = {}
    reporter = get_reporter().child("get-latency")
    for model in models_list:
        results = asyncio.run(_run_all(model))
        all_results[model] = [vars(r) for r in results]
        reporter.step(f"Measured {model}")
        _print_results_table(model, prompt_file, max_new_tokens, results)

    _save_latency_json(output_json, all_results)
    return all_results


def _run_latency_benchmark_openai(
    models_list: List[str],
    prompt: str,
    prompt_file: str,
    output_json: str,
    batch_sizes_list: List[int],
    max_new_tokens: int,
    seed: int,
    model_config_path: str,
) -> Dict[str, Any]:
    AsyncOpenAI = _require_openai_dep()
    routes = _load_model_routes(model_config_path)
    if not routes:
        raise ValueError(
            f"No OpenAI-compatible model routes found in model config: {model_config_path}"
        )

    resolved: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for model in models_list:
        route = _resolve_route(model, routes)
        if route is None:
            missing.append(model)
        else:
            resolved[model] = route
    if missing:
        raise ValueError(
            "Could not resolve latency models via model config aliases: "
            + ", ".join(missing)
        )

    async def _run_all(model_alias: str, route: Dict[str, Any]):
        client = AsyncOpenAI(
            api_key=route["api_key"],
            base_url=route["base_url"],
            timeout=1800.0,
        )
        try:
            warm_tokens = min(8, max_new_tokens)
            await _stream_one_openai(
                client=client,
                request_model=route["request_model"],
                prompt=prompt,
                max_new_tokens=warm_tokens,
                seed=seed,
            )

            results = []
            for bs in batch_sizes_list:
                res = await measure_batch_openai(
                    client=client,
                    request_model=route["request_model"],
                    prompt_text=prompt,
                    batch_size=bs,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                results.append(res)
            return results
        finally:
            try:
                await client.close()
            except Exception:
                pass

    all_results: Dict[str, Any] = {}
    reporter = get_reporter().child("get-latency")
    for model in models_list:
        route = resolved[model]
        reporter.detail(
            f"Measuring via OpenAI endpoint for model '{model}' "
            f"using request model '{route['request_model']}' @ {route['base_url']}"
        )
        results = asyncio.run(_run_all(model, route))
        all_results[model] = [vars(r) for r in results]
        reporter.step(f"Measured {model}")
        _print_results_table(model, prompt_file, max_new_tokens, results)

    _save_latency_json(output_json, all_results)
    return all_results


def run_latency_benchmark(
    models: Union[str, Sequence[str]],
    output_json: str,
    prompt_file: str = "data/prompts/long_text.txt",
    batch_size: int = 1,
    batch_sizes: Optional[Union[str, Sequence[int]]] = None,
    max_new_tokens: int = 1024,
    dtype: str = "auto",
    tp: int = 1,
    gpu_mem_util: float = 0.90,
    seed: int = 0,
    model_config_path: Optional[str] = None,
    backend: str = "auto",
) -> dict:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError(f"{prompt_file} is empty.")

    models_list = _normalize_models(models)
    if not models_list:
        raise ValueError("No models provided for latency benchmarking.")

    batch_sizes_list = _normalize_batch_sizes(batch_sizes, batch_size)
    backend_choice = (backend or "auto").lower()
    if backend_choice not in {"auto", "vllm", "openai"}:
        raise ValueError("backend must be one of: auto, vllm, openai")

    if backend_choice != "openai":
        _configure_cuda_multiprocessing()

    if backend_choice == "openai":
        if not model_config_path:
            raise ValueError("model_config_path is required for backend=openai")
        return _run_latency_benchmark_openai(
            models_list=models_list,
            prompt=prompt,
            prompt_file=prompt_file,
            output_json=output_json,
            batch_sizes_list=batch_sizes_list,
            max_new_tokens=max_new_tokens,
            seed=seed,
            model_config_path=model_config_path,
        )

    if backend_choice == "auto" and model_config_path and not _cuda_available():
        get_reporter().child("get-latency").warn(
            "CUDA device unavailable; using OpenAI-compatible backend "
            f"with model config: {model_config_path}"
        )
        return _run_latency_benchmark_openai(
            models_list=models_list,
            prompt=prompt,
            prompt_file=prompt_file,
            output_json=output_json,
            batch_sizes_list=batch_sizes_list,
            max_new_tokens=max_new_tokens,
            seed=seed,
            model_config_path=model_config_path,
        )

    try:
        return _run_latency_benchmark_vllm(
            models_list=models_list,
            prompt=prompt,
            prompt_file=prompt_file,
            output_json=output_json,
            batch_sizes_list=batch_sizes_list,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            tp=tp,
            gpu_mem_util=gpu_mem_util,
            seed=seed,
        )
    except Exception:
        if backend_choice == "auto" and model_config_path:
            get_reporter().child("get-latency").warn(
                "Local vLLM latency run failed; retrying with OpenAI-compatible "
                f"backend via model config: {model_config_path}"
            )
            return _run_latency_benchmark_openai(
                models_list=models_list,
                prompt=prompt,
                prompt_file=prompt_file,
                output_json=output_json,
                batch_sizes_list=batch_sizes_list,
                max_new_tokens=max_new_tokens,
                seed=seed,
                model_config_path=model_config_path,
            )
        raise
