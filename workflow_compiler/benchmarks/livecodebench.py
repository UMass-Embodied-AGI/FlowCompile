import asyncio
import json
import os
import multiprocessing
import threading
import time
import base64
import zlib
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from workflow_compiler.benchmarks.benchmark import BaseBenchmark
from workflow_compiler.benchmarks.registry import register_benchmark
from workflow_compiler.core.logs import logger
import sys
sys.path.append("..")
sys.path.append("benchmarks")
# ensure lcb_runner is installed
from workflow_compiler.core.utils.lcb_runner import run_test

# Key functions copied from LiveCodeBench official evaluation
#sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    try:
        res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)
    except Exception as e:
        # Catch any unhandled exception and store error result
        in_outs = json.loads(sample["input_output"])
        result.append([-4 for _ in range(len(in_outs["inputs"]))])
        metadata_list.append({
            "error_code": -4,
            "error_message": f"Unexpected error in _temp_run: {str(e)}"
        })

def check_correctness(sample, generation, timeout, debug=True):
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
        if debug:
            logger.warning(
                f"Global timeout: {sample.get('question_id', 'unknown')}"
            )

    return result[0], metadata_list[0]

def evaluate_generations_by_problem(args):
    problem_generations, sample, debug, timeout = args
    res = []
    metadata = []
    for generation in problem_generations:
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness(
                sample, generation, timeout=timeout, debug=debug
            )
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        except Exception as e:
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            res.append(curr_res)
            metadata.append(curr_metadata)
    return res, metadata

@register_benchmark()
class LiveCodeBench(BaseBenchmark):
    BENCHMARK_NAME = "LiveCodeBench"
    ALIASES = ["livecodebench", "LiveCodeBench", "LIVECODEBENCH"]
    WORKFLOW_TYPE = "livecodebench"
    METRIC_NAME = "pass_at_1"
    DEFAULT_SPLIT_PATHS = {
        "validate": "data/ours/livecodebench_validate.jsonl",
        "test": "data/ours/livecodebench_test.jsonl",
    }
    DEFAULT_INIT_KWARGS = {
        "timeout": 6,
        "entry_point_file": "data/ours/livecodebench_public_test.jsonl",
    }

    def __init__(self, name: str, file_path: str, log_path: str, timeout: int = 6, entry_point_file: str = None):
        super().__init__(name, file_path, log_path)
        self.timeout = timeout
        self.num_process_evaluate = min(16, os.cpu_count() or 4)
        self.entry_point_file = entry_point_file
        self.entry_point_map = None  # Will be loaded lazily

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout):
        result = []
        exception_occurred = []
        stop_event = threading.Event()

        def target():
            try:
                return_value = func(*args)
                result.append(return_value)
            except Exception as e:
                exception_occurred.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")
        if exception_occurred:
            raise exception_occurred[0]
        return result[0] if result else None
    def parse_code(self, prediction):
        prediction = prediction.split("```python")[-1]
        prediction = prediction.split("```")[0]
        return prediction
    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        """Load data from JSONL and convert to LiveCodeBench evaluation format."""
        start_time = time.time()
        
        # Create cache file path based on the input files
        cache_dir = os.path.join(os.path.dirname(self.file_path), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache key from file paths and their modification times
        file_mtime = os.path.getmtime(self.file_path)
        entry_point_mtime = os.path.getmtime(self.entry_point_file) if self.entry_point_file and os.path.exists(self.entry_point_file) else 0
        cache_key = f"{os.path.basename(self.file_path)}_{file_mtime}_{entry_point_mtime}"
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                logger.info(f"Loading data from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    processed_data = cached_data['processed_data']
                    self.entry_point_map = cached_data.get('entry_point_map')
                
                elapsed_time = time.time() - start_time
                logger.info(f"Loaded {len(processed_data)} problems from cache in {elapsed_time:.2f} seconds")
                
                if specific_indices is not None:
                    return [processed_data[i] for i in specific_indices if i < len(processed_data)]
                return processed_data
            except Exception as e:
                logger.warning(f"Failed to load cache, will regenerate: {e}")
        
        # Load entry_point mapping from public_test file if provided
        if self.entry_point_file and self.entry_point_map is None:
            self.entry_point_map = {}
            async with aiofiles.open(self.entry_point_file, mode="r", encoding="utf-8") as file:
                async for line in file:
                    entry_data = json.loads(line)
                    self.entry_point_map[entry_data["question_id"]] = entry_data["entry_point"]
            logger.info(f"Loaded {len(self.entry_point_map)} entry points from {self.entry_point_file}")
        
        raw_data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                raw_data.append(json.loads(line))
        
        # Convert to evaluation format
        processed_data = []
        for item in tqdm(raw_data, desc="Processing LiveCodeBench data"):
            try:
                # Handle private test cases (only use private test cases for evaluation)
                if "private_test_cases" in item:
                    try:
                        private_tests = json.loads(item["private_test_cases"])
                    except:
                        private_tests = json.loads(
                            pickle.loads(
                                zlib.decompress(
                                    base64.b64decode(item["private_test_cases"].encode("utf-8"))
                                )
                            )
                        )
                elif "test" in item:
                    # For public test format
                    private_tests = item["test"]
                else:
                    raise ValueError("No test cases found in item")
                
                # Extract entry_point - prioritize entry_point_map, then direct field, then metadata
                question_id = item['question_id']
                if self.entry_point_map and question_id in self.entry_point_map:
                    entry_point = self.entry_point_map[question_id]
                elif "entry_point" in item:
                    entry_point = item["entry_point"]
                elif "metadata" in item and item["metadata"]:
                    metadata = json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]
                    if "func_name" in metadata and metadata["func_name"]:
                        entry_point = metadata["func_name"]
                    else:
                        raise KeyError(f"No entry_point found for question_id {question_id}")
                else:
                    raise KeyError(f"No entry_point found for question_id {question_id}")
                
                # Build evaluation sample
                # Only set fn_name for call-based problems (not stdio)
                # wrapped_function indicates stdio problem
                fn_name = None if entry_point == "wrapped_function" else entry_point
                
                processed_item = {
                    "question": item.get("question_content", item.get("question", "")),
                    "input_output": json.dumps({
                        "inputs": [t["input"] for t in private_tests],
                        "outputs": [t["output"] for t in private_tests],
                        "fn_name": fn_name
                    }),
                    "question_id": item['question_id'],
                    "canonical_solution": item.get("starter_code", ""),
                    "entry_point": entry_point,
                    "metadata": {
                        "platform": item.get("platform", "unknown"),
                        "original_data": item  # Keep original data
                    }
                }
                processed_data.append(processed_item)   
            
            except KeyError as e:
                if "No entry_point found" in str(e):
                    logger.warning(f"Skipping question {item.get('question_id', 'unknown')}: {str(e)}")
                    continue
                else:
                    logger.error(f"Error processing data: {str(e)}")
                    continue
            except Exception as e:
                logger.error(f"Error processing data: {str(e)}")
                continue
        
        # Save to cache
        try:
            cache_data = {
                'processed_data': processed_data,
                'entry_point_map': self.entry_point_map
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved processed data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
        
        if specific_indices is not None:
            processed_data_filtered = [processed_data[i] for i in specific_indices if i < len(processed_data)]
            elapsed_time = time.time() - start_time
            logger.info(f"Loaded {len(processed_data)} problems from {self.file_path} in {elapsed_time:.2f} seconds")
            return processed_data_filtered
        
        elapsed_time = time.time() - start_time
        logger.info(f"Loaded {len(processed_data)} problems from {self.file_path} in {elapsed_time:.2f} seconds")
        return processed_data

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, agent: Callable, query: dict) -> str:
        return await asyncio.wait_for(agent(query), timeout=3600)

    async def evaluate_problem(self, problem: dict, agent: Callable, save_path: str = None) -> Tuple[str, str, str, float, Dict]:
        question = problem["question"]
        question_id = problem["question_id"]
        
        try:
            logger.info(
                f"Start evaluating LiveCodeBench problem: {question_id}"
            )
            
            # Generate code - pass the full problem dict so workflow can access starter_code
            entry_point = problem["entry_point"]
            question_id = problem["question_id"]
            logger.info(f"entry_point: {entry_point}")
            prediction = await self._generate_output(agent, problem)
            logger.info(
                f"Finished code generation, task: {question_id}"
            )
            prediction = self.parse_code(prediction)
            # Use LiveCodeBench evaluation logic
            # fn_name is already correctly set in load_data (None for stdio, function name for call-based)
            in_outs = json.loads(problem["input_output"])

            sample = {
                "question": question,
                "input_output": json.dumps(in_outs),
                "question_id": question_id,
            }
            # logger.info(f"Start evaluating sample {sample['input_output']}")
            
            # Evaluate in a multiprocessing environment
            args = ([prediction], sample, False, self.timeout)
            loop = asyncio.get_running_loop()
            with ProcessPoolExecutor(max_workers=1) as executor:
                results, metadata = await loop.run_in_executor(
                    executor, evaluate_generations_by_problem, args
                )
            
            # Parse results
            logger.info(f"Test results: {results}")
            test_results = results[0]  # Take all test case results of the first (and only) generation
            test_metadata = metadata[0]
            passed = all(r == 1 for r in test_results)
            score = 1.0 if passed else 0.0
            
            # Build evaluation details
            evaluation_details = {
                "question_id": question_id,
                "test_results": test_results,
                "metadata": test_metadata,
                "execution_success": passed,
                "platform": problem.get("metadata", {}).get("platform", "unknown")
            }
            
            # Build expected output for logging
            expected_output = {
                "question_id": question_id,
                "platform": evaluation_details["platform"],
                "canonical_solution": problem.get("canonical_solution", "")
            }

            # Log failures
            if not passed:
                self.log_mismatch(
                    problem=question,
                    expected_output=json.dumps(expected_output),
                    prediction=prediction,
                    extracted_output=prediction,
                    extract_answer_code="N/A"
                )
                logger.warning(f"Task failed: {question_id}, score: {score}")
            else:
                logger.info(f"Task succeeded: {question_id}, score: {score}")

            result = (question, prediction, json.dumps(expected_output), score, evaluation_details)
            
            # Save results
            if save_path:
                async with aiofiles.open(save_path, mode="a", encoding="utf-8") as file:
                    await file.write(json.dumps(result) + "\n")
            
            return result

        except asyncio.TimeoutError:
            logger.error(f"Code generation timeout: {question_id}")
            evaluation_details = {"question_id": question_id, "error": "Timeout"}
            return (question, "Timeout", "", 0.0, evaluation_details)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Evaluation error: {question_id}, error: {e}")
            evaluation_details = {"question_id": question_id, "error": str(e)}
            return (
                question,
                f"Evaluation error: {str(e)}",
                "",
                0.0,
                evaluation_details,
            )

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        return 0.0, ""

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "evaluation_details"]

    @classmethod
    def score_from_result(cls, result: Tuple[Any, ...]) -> float:
        # format: (question, prediction, expected_output, score, evaluation_details)
        return float(result[-2])

    @classmethod
    def result_key(cls, result: Tuple[Any, ...]) -> str:
        details = result[-1] if result else None
        if isinstance(details, dict) and details.get("question_id") is not None:
            return str(details["question_id"])
        return str(result[0]) if result else ""

    @classmethod
    def trace_key(cls, trace: dict) -> str:
        orig = trace.get("metadata", {}).get("original_sample", {}) or {}
        qid = trace.get("question_id") or orig.get("question_id") or orig.get("id") or orig.get("_id")
        return str(qid) if qid is not None else ""

    async def run_baseline_with_load_data(self, agent: Callable, past_data_path: str = None, max_concurrent_tasks: int = 10):
        all_data = await self.load_data()
        
        if not past_data_path:
            past_data_path = os.path.join(self.log_path, f"{self.name}_results.jsonl")
        
        # Load past results
        past_results = {}
        if os.path.exists(past_data_path):
            async with aiofiles.open(past_data_path, mode="r", encoding="utf-8") as file:
                async for line in file:
                    try:
                        result = json.loads(line)
                        # Use question text as key
                        past_results[result[0]] = result
                    except:
                        continue

        # Filter new problems
        new_data = [p for p in all_data if p["question"] not in past_results]
        
        if not new_data:
            logger.info("All problems have been evaluated")
            return None

        logger.info(
            f"{len(new_data)} new problems to evaluate out of {len(all_data)} total"
        )

        # Evaluate new problems
        new_results = await self.evaluate_all_problems(
            new_data, agent, save_path=past_data_path, max_concurrent_tasks=max_concurrent_tasks
        )
        
        # Merge results
        all_results = list(past_results.values()) + new_results
        
        # Save final results
        columns = self.get_result_columns()
        average_score = self.save_results_to_csv(all_results, columns)
        
        logger.info(f"{self.name} dataset average score: {average_score:.5f}")
        return average_score
