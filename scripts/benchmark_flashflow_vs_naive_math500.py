from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import gc
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
FLASHFLOW_ROOT = REPO_ROOT / "3rdparty" / "flashflow"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FLASHFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(FLASHFLOW_ROOT))

from flashflow.backends.vllm import VLLMBackend
from flashflow.types import AliasInfo
from workflow_compiler.core.llm.config import build_setting
from workflow_compiler.dsl.executor import DslExecutor
import workflow_compiler.dsl.executor as executor_module
from workflow_compiler.runtime.export import export_flashflow_dag, write_flashflow_dag
from workflow_compiler.workflows.dsl_registry import get_workflow_module


DATASET_PATH = REPO_ROOT / "data" / "math500_test.jsonl"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
TMP_DIR = REPO_ROOT / "temp" / "benchmark_outputs"
FLASHFLOW_PORT = 8023
MAX_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 3600

AGENT_SETTINGS: Dict[str, str] = {
    "generate_solver": "qwen3-4b_budget_500",
    "detailed_solver": "qwen3-1.7b_budget_1000",
    "refine_solver": "qwen3-1.7b_budget_1000",
    "programmer": "qwen3-8b_budget_100",
    "sc_ensemble": "qwen3-8b_budget_100",
}

VLLM_ARGS: Dict[str, Any] = {
    "tensor_parallel_size": 1,
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.8,
    "max_num_seqs": 64,
}


def _load_jsonl_samples(path: Path, limit: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if len(samples) >= limit:
                break
            samples.append(json.loads(line))
    return samples


def _load_model_payload(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    models = payload.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError(f"Invalid model config: missing models map in {path}")
    return payload


def _build_selected_config() -> Dict[str, Any]:
    return {
        "config_id": "math500_flashflow_benchmark_mixed_qwen",
        "workflow_type": "math",
        "agents": {
            agent_name: {"setting": setting}
            for agent_name, setting in AGENT_SETTINGS.items()
        },
    }


def _build_flashflow_dag(output_path: Path) -> Path:
    compiled_payload = {
        "workflow_type": "math",
        "configs": [_build_selected_config()],
        "all_configs": [_build_selected_config()],
    }
    exported_dag, _ = export_flashflow_dag(
        compiled_payload=compiled_payload,
        model_config=_load_model_payload(MODEL_CONFIG_PATH),
        workflow_type="math",
        config_id="math500_flashflow_benchmark_mixed_qwen",
    )
    write_flashflow_dag(output_path, exported_dag)
    return output_path


def _sum_usage(usage: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    input_tokens = sum(int(item.get("input", 0)) for item in usage.values())
    output_tokens = sum(int(item.get("output", 0)) for item in usage.values())
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class UsageLedger:
    def __init__(self) -> None:
        self._usage: Dict[str, Dict[str, int]] = {}

    def reset(self) -> None:
        self._usage = {}

    def add(self, model_name: str, input_tokens: int, output_tokens: int) -> None:
        slot = self._usage.setdefault(model_name, {"input": 0, "output": 0})
        slot["input"] += int(input_tokens)
        slot["output"] += int(output_tokens)

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return copy.deepcopy(self._usage)


class NoopUsageTracker:
    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, int]:
        return {
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens) + int(output_tokens),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {}

    async def aclose(self) -> None:
        return None


class FlashFlowHttpLLM:
    def __init__(self, model_name: str, base_url: str) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
        self.usage_tracker = NoopUsageTracker()
        self.config = SimpleNamespace(model=model_name, api_type="openai", temperature=1, top_p=1, raw={})

    async def __call__(self, prompt: str, return_io_tokens: bool = False):
        return await self.call_with_thinking_budget(prompt, None, return_io_tokens=return_io_tokens)

    async def call_with_thinking_budget(
        self,
        prompt: str,
        thinking_budget: Any,
        return_io_tokens: bool = False,
    ):
        alias = build_setting(self.model_name, thinking_budget)
        payload = {
            "model": alias,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1,
            "top_p": 1,
            "max_tokens": MAX_TOKENS,
        }
        response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        text = body["choices"][0]["message"]["content"]
        if return_io_tokens:
            return text, input_tokens, output_tokens
        return text

    def get_usage_summary(self) -> Dict[str, Any]:
        return {}

    async def aclose(self) -> None:
        await self.client.aclose()


class NaiveDirectVllmLLM:
    def __init__(self, model_name: str, model_cfg: Dict[str, Any], ledger: UsageLedger) -> None:
        self.model_name = model_name
        self.model_cfg = dict(model_cfg)
        self.ledger = ledger
        self.usage_tracker = NoopUsageTracker()
        self.config = SimpleNamespace(model=model_name, api_type="openai", temperature=1, top_p=1, raw=self.model_cfg)

    async def __call__(self, prompt: str, return_io_tokens: bool = False):
        return await self.call_with_thinking_budget(prompt, None, return_io_tokens=return_io_tokens)

    async def call_with_thinking_budget(
        self,
        prompt: str,
        thinking_budget: Any,
        return_io_tokens: bool = False,
    ):
        alias = build_setting(self.model_name, thinking_budget)
        alias_info = AliasInfo(
            model_alias=alias,
            base_model=self.model_name,
            budget=thinking_budget,
            backend="vllm",
            default_thinking_strategy="vllm_plugin",
            metadata={},
        )
        backend = VLLMBackend(
            self.model_name,
            {
                "backend": "vllm",
                "artifact_id": self.model_cfg.get("hf_model_name") or self.model_cfg.get("model") or self.model_name,
                "tokenizer": self.model_cfg.get("hf_model_name") or self.model_cfg.get("tokenizer") or self.model_name,
            },
            dict(VLLM_ARGS, enable_sleep_mode=False),
        )
        try:
            await backend.initialize()
            result = await backend.generate_chat(
                messages=[{"role": "user", "content": prompt}],
                alias_info=alias_info,
                request={"temperature": 1, "top_p": 1, "max_tokens": MAX_TOKENS},
            )
        finally:
            await _shutdown_backend(backend)
        self.ledger.add(result.model_name, result.input_tokens, result.output_tokens)
        if return_io_tokens:
            return result.text, result.input_tokens, result.output_tokens
        return result.text

    def get_usage_summary(self) -> Dict[str, Any]:
        return self.ledger.snapshot()

    async def aclose(self) -> None:
        return None


async def _shutdown_backend(backend: VLLMBackend) -> None:
    llm = getattr(backend, "_llm", None)
    if llm is None:
        return
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        await asyncio.to_thread(shutdown)
    backend._llm = None
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


@contextlib.contextmanager
def _patch_llm_factory(factory):
    original = executor_module.create_llm_instance
    executor_module.create_llm_instance = factory
    try:
        yield
    finally:
        executor_module.create_llm_instance = original


def _build_executor() -> DslExecutor:
    workflow_module = get_workflow_module("math")
    spec = workflow_module.compile()
    return DslExecutor(spec, "math", {"agents": _build_selected_config()["agents"]})


async def _run_executor_samples(executor: DslExecutor, samples: List[Dict[str, Any]], on_before_sample, on_after_sample):
    records: List[Dict[str, Any]] = []
    try:
        for idx, sample in enumerate(samples, 1):
            await on_before_sample(idx, sample)
            started = time.perf_counter()
            outputs, steps, _ = await executor.run({"query": sample})
            elapsed = time.perf_counter() - started
            usage = await on_after_sample(idx, sample)
            record = {
                "sample_index": idx,
                "query_id": sample.get("unique_id") or sample.get("id") or str(idx),
                "latency_seconds": elapsed,
                "usage_by_model": usage,
                "usage_totals": _sum_usage(usage),
                "final_answer": outputs.get("final_answer"),
                "step_count": len(steps),
            }
            records.append(record)
            print(
                f"[sample {idx:02d}] latency={elapsed:.2f}s "
                f"input={record['usage_totals']['input_tokens']} "
                f"output={record['usage_totals']['output_tokens']}",
                flush=True,
            )
    finally:
        await executor.aclose()
    return records


async def _flashflow_admin_reset(client: httpx.AsyncClient, base_url: str) -> None:
    response = await client.post(f"{base_url}/flashflow/token_usage/reset")
    response.raise_for_status()


async def _flashflow_admin_get(client: httpx.AsyncClient, base_url: str) -> Dict[str, Dict[str, int]]:
    response = await client.get(f"{base_url}/flashflow/token_usage")
    response.raise_for_status()
    return response.json().get("data") or {}


async def _wait_for_flashflow(
    base_url: str,
    process: subprocess.Popen[Any],
    timeout_seconds: float = 1800.0,
) -> None:
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"FlashFlow exited early with status {process.returncode}.")
            try:
                response = await client.get(f"{base_url}/models")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            if time.perf_counter() - started > timeout_seconds:
                raise TimeoutError(f"FlashFlow did not become ready within {timeout_seconds} seconds.")
            await asyncio.sleep(2.0)


@dataclass
class FlashFlowServerHandle:
    process: subprocess.Popen[Any]
    log_path: Path
    startup_seconds: float
    base_url: str


async def _start_flashflow_server(dag_path: Path, port: int) -> FlashFlowServerHandle:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TMP_DIR / f"flashflow_server_{port}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{FLASHFLOW_ROOT}"
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "flashflow.cli",
                "serve",
                str(dag_path),
                "--port",
                str(port),
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                str(VLLM_ARGS["gpu_memory_utilization"]),
                "--max-num-seqs",
                str(VLLM_ARGS["max_num_seqs"]),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        await _wait_for_flashflow(base_url, process)
    except Exception:
        _stop_process_tree(process)
        raise
    startup_seconds = time.perf_counter() - started
    return FlashFlowServerHandle(
        process=process,
        log_path=log_path,
        startup_seconds=startup_seconds,
        base_url=base_url,
    )


def _stop_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(Exception):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(Exception):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        process.wait(timeout=10)


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_latency = sum(float(item["latency_seconds"]) for item in records)
    total_input = sum(int(item["usage_totals"]["input_tokens"]) for item in records)
    total_output = sum(int(item["usage_totals"]["output_tokens"]) for item in records)
    return {
        "samples": len(records),
        "total_latency_seconds": total_latency,
        "avg_latency_seconds": (total_latency / len(records)) if records else 0.0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


async def benchmark_flashflow(samples: List[Dict[str, Any]], dag_path: Path, port: int) -> Dict[str, Any]:
    server = await _start_flashflow_server(dag_path, port)
    client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        with _patch_llm_factory(lambda model_name: FlashFlowHttpLLM(model_name, server.base_url)):
            executor = _build_executor()
            records = await _run_executor_samples(
                executor,
                samples,
                on_before_sample=lambda idx, sample: _flashflow_admin_reset(client, server.base_url),
                on_after_sample=lambda idx, sample: _flashflow_admin_get(client, server.base_url),
            )
    finally:
        await client.aclose()
        _stop_process_tree(server.process)
    summary = _summarize(records)
    summary["startup_seconds"] = server.startup_seconds
    summary["total_with_startup_seconds"] = server.startup_seconds + summary["total_latency_seconds"]
    summary["server_log"] = str(server.log_path)
    return {
        "summary": summary,
        "samples": records,
    }


async def benchmark_naive(samples: List[Dict[str, Any]], model_payload: Dict[str, Any]) -> Dict[str, Any]:
    models_payload = model_payload.get("models") or {}
    ledger = UsageLedger()

    def factory(model_name: str):
        model_cfg = models_payload.get(model_name)
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Missing model config for {model_name}")
        return NaiveDirectVllmLLM(model_name, model_cfg, ledger)

    with _patch_llm_factory(factory):
        executor = _build_executor()
        records = await _run_executor_samples(
            executor,
            samples,
            on_before_sample=lambda idx, sample: asyncio.to_thread(ledger.reset),
            on_after_sample=lambda idx, sample: asyncio.to_thread(ledger.snapshot),
        )
    return {
        "summary": _summarize(records),
        "samples": records,
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _print_final_report(result: Dict[str, Any]) -> None:
    flashflow_summary = result["flashflow"]["summary"]
    naive_summary = result["naive"]["summary"]

    steady_speedup = _ratio(
        naive_summary["avg_latency_seconds"],
        flashflow_summary["avg_latency_seconds"],
    )
    end_to_end_speedup = _ratio(
        naive_summary["total_latency_seconds"],
        flashflow_summary["total_with_startup_seconds"],
    )

    print("\n=== Benchmark Summary ===", flush=True)
    print(
        "FlashFlow steady-state: "
        f"{flashflow_summary['total_latency_seconds']:.2f}s total, "
        f"{flashflow_summary['avg_latency_seconds']:.2f}s/sample, "
        f"{flashflow_summary['total_input_tokens']} input, "
        f"{flashflow_summary['total_output_tokens']} output",
        flush=True,
    )
    print(
        "FlashFlow with startup: "
        f"{flashflow_summary['total_with_startup_seconds']:.2f}s total "
        f"(startup {flashflow_summary['startup_seconds']:.2f}s)",
        flush=True,
    )
    print(
        "Naive cold-load vLLM: "
        f"{naive_summary['total_latency_seconds']:.2f}s total, "
        f"{naive_summary['avg_latency_seconds']:.2f}s/sample, "
        f"{naive_summary['total_input_tokens']} input, "
        f"{naive_summary['total_output_tokens']} output",
        flush=True,
    )
    print(
        "Speedup: "
        f"{steady_speedup:.2f}x steady-state, "
        f"{end_to_end_speedup:.2f}x end-to-end including FlashFlow startup",
        flush=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FlashFlow vs naive one-by-one vLLM on the first math500 test samples."
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--port", type=int, default=FLASHFLOW_PORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=TMP_DIR / "flashflow_vs_naive_math500_10samples.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    samples = _load_jsonl_samples(DATASET_PATH, args.samples)
    model_payload = _load_model_payload(MODEL_CONFIG_PATH)
    dag_path = _build_flashflow_dag(TMP_DIR / "math500_mixed_qwen_benchmark_dag.json")

    print(f"Loaded {len(samples)} math500 samples from {DATASET_PATH}", flush=True)
    print(f"Using exported DAG: {dag_path}", flush=True)
    print("Benchmarking FlashFlow...", flush=True)
    flashflow_result = await benchmark_flashflow(samples, dag_path, args.port)
    print("Benchmarking naive direct vLLM...", flush=True)
    naive_result = await benchmark_naive(samples, model_payload)

    result = {
        "setup": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensor_parallel_size": 1,
            "workflow_type": "math",
            "dataset": str(DATASET_PATH),
            "sample_count": len(samples),
            "agent_settings": dict(AGENT_SETTINGS),
            "max_tokens": MAX_TOKENS,
            "flashflow_port": args.port,
        },
        "flashflow": flashflow_result,
        "naive": naive_result,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(flashflow_result["summary"], indent=2), flush=True)
    print(json.dumps(naive_result["summary"], indent=2), flush=True)
    _print_final_report(result)
    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
