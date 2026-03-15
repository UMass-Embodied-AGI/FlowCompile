#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark token usage and accuracy of models with different reasoning budgets for sub-agents.

USAGE:
    flowcompile profile --experiment-id <experiment_id> [OPTIONS]

EXAMPLES:
    # Benchmark MATH workflow (default models and budgets)
    flowcompile profile --experiment-id 1208_math500
    
    # Benchmark HotpotQA workflow
    flowcompile profile --experiment-id 1221_hotpotqa
    
    # Benchmark LiveCodeBench workflow
    flowcompile profile --experiment-id 1222_livecodebench
    
    # Override models
    flowcompile profile --experiment-id 1208_math500 --models qwen3-4b qwen3-8b
    
    # Limit samples for testing
    flowcompile profile --experiment-id 1208_math500 --max-samples 10

CONFIGURATION:
    - Experiment ID automatically determines:
      * WORKFLOW_TYPE: "math", "hotpotqa", or "livecodebench"
      * TRAINING_DATA_PATH: Path to trace_training_data.json from experiment results
      * OUTPUT_DIR: Results directory
      * SEARCH_BUDGETS: Thinking budgets used for profiling
      * AGENT_NAMES: Agents to benchmark
    
    - Training data must exist at: results/<experiment_id>/data/*/trace_training_data.json
    - Results will be saved to: results/<experiment_id>/benchmark_<timestamp>/

WHAT THIS SCRIPT DOES:
    1. Loads training data from trace_training_data.json
    2. Tests multiple models with different thinking budgets
    3. Evaluates each configuration N times per sample
    4. Uses gpt-oss-120b as judge to compare outputs with ground truth
    5. Tracks token usage (input/output tokens) and accuracy
    6. Saves results to a comprehensive JSON file
"""

import asyncio
import json
import random
import re
import string
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from tqdm.asyncio import tqdm

from workflow_compiler.compiler.judge_types import JudgeContext, JudgeResult
from workflow_compiler.core.llm.client import AsyncLLM, create_llm_instance
from workflow_compiler.core.workflow.operators import run_code
from workflow_compiler.benchmarks.livecodebench import evaluate_generations_by_problem
from workflow_compiler.core.data_paths import resolve_existing_path
from concurrent.futures import ProcessPoolExecutor
from workflow_compiler.workflows.dsl_registry import get_workflow_module
from workflow_compiler.integration.openclaw import normalize_openclaw_agent_policies

GLOBAL_DEFAULT_SEARCH_BUDGETS = (10, 200, 400, 800, 1000, 1500, 2000, 3000, 4000, 5000)
VALID_JUDGE_POLICY_MODES = {"semantic_llm"}


def normalize_judge_policies(raw: Any) -> Dict[str, Dict[str, Any]]:
    if raw in (None, "", {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("judge_policies must be a mapping of agent -> policy")

    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_agent, raw_policy in raw.items():
        agent_name = str(raw_agent or "").strip()
        if not agent_name:
            raise ValueError("judge_policies contains an empty agent name")
        if not isinstance(raw_policy, dict):
            raise ValueError(f"judge_policies[{agent_name!r}] must be a mapping")

        mode = str(raw_policy.get("mode") or "").strip().lower()
        if mode not in VALID_JUDGE_POLICY_MODES:
            raise ValueError(
                f"judge_policies[{agent_name!r}].mode must be one of: "
                f"{', '.join(sorted(VALID_JUDGE_POLICY_MODES))}"
            )

        prompt = str(raw_policy.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"judge_policies[{agent_name!r}].prompt is required")

        normalized[agent_name] = {
            "mode": mode,
            "prompt": prompt,
        }
    return normalized


def _run_code_in_subprocess(code: str, result_queue) -> None:
    """Top-level subprocess target for programmer-agent code evaluation."""
    try:
        status, output = run_code(code)
        result_queue.put(("success", status, output))
    except Exception as e:
        result_queue.put(("error", str(e), ""))


def get_experiment_config(
    experiment_id: str,
    search_budgets: Optional[List[Any]] = None,
    workflow_type: Optional[str] = None,
    training_data_path: Optional[str] = None,
    experiment_root: Optional[str] = None,
    openclaw_lobster_workflow_file: Optional[str] = None,
    openclaw_agent_policies: Optional[Dict[str, Any]] = None,
    judge_policies: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get benchmark configuration based on experiment ID.
    
    This function maps experiment IDs to their corresponding:
    - TRAINING_DATA_PATH: Path to the training data file
    - OUTPUT_DIR: Output directory for results
    - WORKFLOW_TYPE: Either "math" or "hotpotqa"
    - SEARCH_BUDGETS: List of thinking budgets to test
    - AGENT_NAMES: List of agents to benchmark
    
    Args:
        experiment_id: Experiment ID (e.g., "1208_math500", "1221_hotpotqa")
    
    Returns:
        Dictionary with configuration parameters
    """
    results_dir = Path(experiment_root) if experiment_root else Path("results") / experiment_id
    profile_dir = results_dir / "01_profile"
    
    if not results_dir.exists() and not training_data_path:
        raise FileNotFoundError(f"Experiment directory not found: {results_dir}")
    
    # Explicit training data path has highest precedence.
    training_data_file = None
    if training_data_path:
        candidate = Path(training_data_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Configured training_data_path not found: {candidate}")
        training_data_file = str(candidate)

    if not training_data_file:
        aggregated_candidates = [
            profile_dir / "aggregated_training_data.json",
            results_dir / "data" / "aggregated_training_data.json",
            results_dir / "aggregated_training_data.json",
        ]
        for candidate in aggregated_candidates:
            if candidate.exists():
                training_data_file = str(candidate)
                break

    # Legacy fallback: search for trace_training_data.json under common roots.
    if not training_data_file:
        trace_candidates: List[Path] = []
        for root in (profile_dir, results_dir / "data", results_dir):
            if not root.exists():
                continue
            for trace_file in root.glob("**/trace_training_data.json"):
                if trace_file.is_file():
                    trace_candidates.append(trace_file)
        if trace_candidates:
            training_data_file = str(max(trace_candidates, key=lambda p: p.stat().st_mtime))
    
    if not training_data_file:
        raise FileNotFoundError(
            "No training data found. Expected one of:\n"
            f"  - {profile_dir / 'aggregated_training_data.json'}\n"
            f"  - {results_dir / 'data' / 'aggregated_training_data.json'}\n"
            f"  - {results_dir}/**/trace_training_data.json\n"
            "Please run `flowcompile prepare-data` first."
        )
    
    resolved_workflow_type = str(workflow_type or "").strip().lower()
    if not resolved_workflow_type:
        # Determine workflow type from experiment ID
        # Note: 'math', 'math500', and 'gsm8k' datasets all use workflow_type='math'
        # since they share the same workflow structure.
        if "math" in experiment_id.lower() or "gsm8k" in experiment_id.lower():
            resolved_workflow_type = "math"
        elif "hotpotqa" in experiment_id.lower():
            resolved_workflow_type = "hotpotqa"
        elif "livecodebench" in experiment_id.lower():
            resolved_workflow_type = "livecodebench"
        else:
            # Default fallback
            resolved_workflow_type = "math"

    if resolved_workflow_type == "openclaw_lobster" and not openclaw_lobster_workflow_file:
        raise ValueError(
            "openclaw_lobster profiling requires openclaw_lobster_workflow_file."
        )

    workflow_module = get_workflow_module(
        resolved_workflow_type,
        openclaw_lobster_workflow_file=openclaw_lobster_workflow_file,
    )
    inferred_agents = workflow_module.infer_profiling_agents()
    selected_search_budgets = list(search_budgets) if search_budgets else list(GLOBAL_DEFAULT_SEARCH_BUDGETS)

    return {
        "training_data_path": training_data_file,
        "output_dir": str(profile_dir),
        "workflow_type": resolved_workflow_type,
        "search_budgets": selected_search_budgets,
        "agent_names": inferred_agents,
        "workflow_module": workflow_module,
        "workflow_judges": workflow_module.get_profiling_judges(),
        "openclaw_lobster_workflow_file": openclaw_lobster_workflow_file,
        "openclaw_agent_policies": normalize_openclaw_agent_policies(openclaw_agent_policies),
    }


class BenchmarkConfig:
    """Configuration for the benchmark"""
    
    # Models to benchmark
    MODELS = [
        "qwen3-0.6b",
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-14b",
    ]

    # Thinking budgets for GPT-OSS models (string-based)
    GPT_OSS_BUDGETS = ["low", "medium", "high"]
    
    # Judge model
    JUDGE_MODEL = "gpt-oss-120b"
    
    # Number of repetitions per sample
    N_REPETITIONS = 1
    
    # Maximum concurrent tasks for parallel inference
    MAX_CONCURRENT = 64
    
    # Debug mode: when enabled, disables parallel evaluation and uses sequential mode
    DEBUG = False
    
    # Minimum samples per agent: if an agent has fewer samples, duplicate them until reaching this minimum
    # When data is duplicated, N_REPETITIONS is set to 1 to avoid double repetition
    # Set to None to disable (default)
    MIN_SAMPLES_PER_AGENT = None
    
    # Optional cap on samples per agent for fast smoke testing
    # Set to None to use all available samples
    MAX_SAMPLES_PER_AGENT = None
    
    # Configuration parameters (will be set by initialize_from_experiment_id)
    TRAINING_DATA_PATH = None
    OUTPUT_DIR = None
    SEARCH_BUDGETS = None
    AGENT_NAMES = None
    WORKFLOW_TYPE = None
    WORKFLOW_MODULE = None
    WORKFLOW_JUDGES = None
    OPENCLAW_LOBSTER_WORKFLOW_FILE = None
    OPENCLAW_AGENT_POLICIES = None
    LIVECODEBENCH_VALIDATE_PATH = "data/livecodebench_validate.jsonl"
    LIVECODEBENCH_PUBLIC_TEST_PATH = "data/livecodebench_public_test.jsonl"
    
    @classmethod
    def initialize_from_experiment_id(
        cls,
        experiment_id: str,
        search_budgets: Optional[List[Any]] = None,
        workflow_type: Optional[str] = None,
        training_data_path: Optional[str] = None,
        experiment_root: Optional[str] = None,
        openclaw_lobster_workflow_file: Optional[str] = None,
        openclaw_agent_policies: Optional[Dict[str, Any]] = None,
        judge_policies: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration from experiment ID.
        
        Args:
            experiment_id: Experiment ID (e.g., "1208_math500", "1221_hotpotqa")
        """
        config = get_experiment_config(
            experiment_id,
            search_budgets=search_budgets,
            workflow_type=workflow_type,
            training_data_path=training_data_path,
            experiment_root=experiment_root,
            openclaw_lobster_workflow_file=openclaw_lobster_workflow_file,
            openclaw_agent_policies=openclaw_agent_policies,
            judge_policies=judge_policies,
        )
        cls.TRAINING_DATA_PATH = config["training_data_path"]
        cls.OUTPUT_DIR = config["output_dir"]
        cls.SEARCH_BUDGETS = config["search_budgets"]
        cls.AGENT_NAMES = config["agent_names"]
        cls.WORKFLOW_TYPE = config["workflow_type"]
        cls.WORKFLOW_MODULE = config.get("workflow_module")
        cls.WORKFLOW_JUDGES = config.get("workflow_judges") or {}
        cls.OPENCLAW_LOBSTER_WORKFLOW_FILE = config.get("openclaw_lobster_workflow_file")
        cls.OPENCLAW_AGENT_POLICIES = config.get("openclaw_agent_policies") or None
        if judge_policies and cls.WORKFLOW_TYPE != "openclaw_lobster":
            print(
                "Warning: judge_policies are ignored for built-in workflows; "
                "define workflow-owned judges in workflow_compiler/workflows/<workflow>/judges.py."
            )
        
        print(f"\n{'='*80}")
        print(f"Configuration loaded for experiment: {experiment_id}")
        print(f"{'='*80}")
        print(f"Workflow Type: {cls.WORKFLOW_TYPE}")
        print(f"Training Data: {cls.TRAINING_DATA_PATH}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print(f"Search Budgets: {cls.SEARCH_BUDGETS}")
        print(f"Agents: {cls.AGENT_NAMES}")
        print(f"{'='*80}\n")


class JudgeEvaluator:
    """Evaluates model outputs using a judge model"""

    @classmethod
    def _openclaw_agent_policies(cls) -> Dict[str, Dict[str, Any]]:
        configured = getattr(BenchmarkConfig, "OPENCLAW_AGENT_POLICIES", None)
        return configured or {}

    @classmethod
    def _workflow_judges(cls) -> Dict[str, Any]:
        configured = getattr(BenchmarkConfig, "WORKFLOW_JUDGES", None)
        return configured or {}
    
    def __init__(self, judge_model: str = BenchmarkConfig.JUDGE_MODEL):
        """Initialize with judge model"""
        self.judge_llm = create_llm_instance(judge_model, endpoint_role="profile")
        self.livecodebench_test_cache = {}  # Cache for LiveCodeBench test cases
        self._load_livecodebench_test_cache()

    async def aclose(self):
        """Close judge LLM client."""
        close_method = getattr(self.judge_llm, "aclose", None)
        if close_method:
            try:
                await close_method()
            except Exception:
                pass
    
    def _load_livecodebench_test_cache(self):
        """
        Load and cache all LiveCodeBench test cases once on initialization.
        This avoids repeated file I/O during evaluation.
        Loads entry_point from livecodebench_public_test.jsonl and test cases from livecodebench_validate.jsonl.
        """
        try:
            # First, load entry_point mappings from public_test file
            entry_point_map = {}
            public_test_path = Path(
                resolve_existing_path(BenchmarkConfig.LIVECODEBENCH_PUBLIC_TEST_PATH)
                or BenchmarkConfig.LIVECODEBENCH_PUBLIC_TEST_PATH
            )
            if public_test_path.exists():
                print(f"Loading entry_point mappings from {public_test_path}...")
                with open(public_test_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            question_id = item.get("question_id")
                            entry_point = item.get("entry_point", "wrapped_function")
                            if question_id:
                                entry_point_map[question_id] = entry_point
                        except Exception as e:
                            continue
                print(f"Loaded {len(entry_point_map)} entry_point mappings")
            else:
                print(f"Warning: entry_point file not found: {public_test_path}")
            
            # Now load test cases from validate file
            test_data_path = Path(
                resolve_existing_path(BenchmarkConfig.LIVECODEBENCH_VALIDATE_PATH)
                or BenchmarkConfig.LIVECODEBENCH_VALIDATE_PATH
            )
            if not test_data_path.exists():
                print(f"Warning: LiveCodeBench test data file not found: {test_data_path}")
                return
            
            print(f"Loading LiveCodeBench test cases from {test_data_path}...")
            count = 0
            with open(test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        question_id = item.get("question_id")
                        
                        if not question_id:
                            continue
                        
                        # Decode private test cases if present
                        test_cases = None
                        if "private_test_cases" in item:
                            try:
                                test_cases = json.loads(item["private_test_cases"])
                            except:
                                # Handle compressed format
                                import base64
                                import zlib
                                import pickle
                                test_cases = json.loads(
                                    pickle.loads(
                                        zlib.decompress(
                                            base64.b64decode(item["private_test_cases"].encode("utf-8"))
                                        )
                                    )
                                )
                        
                        # Get entry_point from the pre-loaded map
                        entry_point = entry_point_map.get(question_id, "wrapped_function")
                        
                        # Store the test data in cache
                        self.livecodebench_test_cache[question_id] = {
                            "test_cases": test_cases,
                            "entry_point": entry_point,
                            "metadata": item.get("metadata"),
                            "question": item.get("question_content", item.get("question", "")),
                        }
                        count += 1
                    except Exception as e:
                        print(f"Warning: Failed to parse line in LiveCodeBench test data: {e}")
                        continue
            
            print(f"Successfully loaded {count} test cases from LiveCodeBench")
            
        except Exception as e:
            print(f"Error loading LiveCodeBench test cache: {e}")
            import traceback
            traceback.print_exc()

    def _get_workflow_judge(self, agent_name: str):
        return self._workflow_judges().get(agent_name)

    @staticmethod
    def _log_failure(agent_name: str, reason_code: str, detail: str = "") -> None:
        message = f"[judge:{reason_code}] agent={agent_name}"
        if detail:
            message = f"{message} {detail}"
        print(message)

    @staticmethod
    def _parse_json_object(text: Any) -> Optional[Dict[str, Any]]:
        if isinstance(text, dict):
            return text
        if not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _normalize_exact_value(value: str) -> str:
        return str(value).strip().lower()

    def _validate_openclaw_payloads(
        self,
        agent_name: str,
        model_output: Any,
        ground_truth: Any,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], List[str]]]:
        policy = self._openclaw_agent_policies().get(agent_name)
        if not policy:
            return None

        predicted_payload = self._parse_json_object(model_output)
        if predicted_payload is None:
            self._log_failure(agent_name, "invalid_json", "model_output must be a JSON object")
            return None

        ground_truth_payload = self._parse_json_object(ground_truth)
        if ground_truth_payload is None:
            self._log_failure(agent_name, "invalid_json", "ground_truth must be a JSON object")
            return None

        required_fields = list(policy.get("required_fields") or [])
        if not required_fields:
            self._log_failure(agent_name, "missing_required_field", "no required field policy configured")
            return None
        for payload_name, payload in (("model_output", predicted_payload), ("ground_truth", ground_truth_payload)):
            for field_name in required_fields:
                if field_name not in payload:
                    self._log_failure(
                        agent_name,
                        "missing_required_field",
                        f"{payload_name} missing field '{field_name}'",
                    )
                    return None
                value = payload.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    self._log_failure(
                        agent_name,
                        "missing_required_field",
                        f"{payload_name}.{field_name} must be a non-empty string",
                    )
                    return None

        return predicted_payload, ground_truth_payload, required_fields
    
    def normalize_answer(self, s: str) -> str:
        """Normalize answer for F1 calculation (from HotpotQA benchmark)"""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def extract_boxed_choice_letter(text: str) -> Optional[str]:
        """Extract ensemble choice letter from text like '\\boxed{A}'."""
        if not text:
            return None
        match = re.search(r'\\boxed\{\s*([A-Za-z])\s*\}', text)
        if match:
            return match.group(1).upper()
        return None
    
    async def evaluate_code_with_private_tests(self, code: str, original_sample: Dict) -> bool:
        """
        Evaluate code using private test cases from LiveCodeBench (cached).
        
        Args:
            code: The generated code to evaluate
            original_sample: Original sample data containing question_id, difficulty, platform
            
        Returns:
            True if all private test cases pass, False otherwise
        """
        try:
            # Extract question_id from original_sample
            if "question_id" not in original_sample:
                print("Warning: No question_id field in original_sample")
                return False
            
            question_id = original_sample["question_id"]
            
            # Look up test cases from cache
            if question_id not in self.livecodebench_test_cache:
                print(f"Warning: No test case found in cache for question_id: {question_id}")
                return False
            
            test_data = self.livecodebench_test_cache[question_id]
            private_tests = test_data.get("test_cases")
            
            if not private_tests:
                print(f"Warning: No private_test_cases for question_id: {question_id}")
                return False
            
            # Determine entry_point (function name)
            entry_point = test_data.get("entry_point", "wrapped_function")
            
            # Build input_output format for evaluation
            fn_name = None if entry_point == "wrapped_function" else entry_point
            
            # Prepare sample for evaluation
            sample = {
                "question": test_data.get("question", ""),
                "input_output": json.dumps({
                    "inputs": [t["input"] for t in private_tests],
                    "outputs": [t["output"] for t in private_tests],
                    "fn_name": fn_name
                }),
                "question_id": question_id,
            }
            
            # Run evaluation using LiveCodeBench evaluator
            args = ([code], sample, False, 30)  # 30 second timeout per test case
            loop = asyncio.get_running_loop()
            with ProcessPoolExecutor(max_workers=1) as executor:
                results, metadata = await loop.run_in_executor(
                    executor, evaluate_generations_by_problem, args
                )
            # Parse results - all test cases must pass (result == 1)
            test_results = results[0]
            all_passed = all(r == 1 for r in test_results)
            
            return all_passed
            
        except Exception as e:
            print(f"Error evaluating code with private tests: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_f1_score(self, ground_truth: str, prediction: str) -> float:
        """Calculate F1 score between ground truth and prediction (from HotpotQA benchmark)"""
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0.0
        recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    async def evaluate_with_f1(self, ground_truth: str, model_output: str, agent_name: str) -> Tuple[bool, float]:
        """Evaluate format_answer agent and return both correctness and F1 score."""
        if not model_output or not model_output.strip():
            return False, 0.0
        f1_score = self.calculate_f1_score(ground_truth, model_output)
        is_correct = f1_score
        return is_correct, f1_score

    def extract_code_from_output(self, output: str) -> Optional[str]:
        """
        Extract Python code from output string.
        Handles both markdown code blocks and plain code.
        
        Args:
            output: String containing code (possibly with markdown formatting)
            
        Returns:
            Extracted code string, or None if no code found
        """
        import re
        
        # Try to extract from markdown code block
        match = re.search(r'```python\s*\n(.*?)\n```', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no markdown block, try to find def solve() directly
        if 'def solve()' in output:
            # Extract from def solve() to the end or until we hit non-code content
            match = re.search(r'(def solve\(\):.*?)(?:\n\n|\nOutput:|\Z)', output, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    async def execute_code_in_subprocess(
        self,
        code: str,
        timeout_seconds: float = 5.0,
    ) -> Tuple[Optional[str], str]:
        try:
            from multiprocessing import Process, Queue

            result_queue = Queue()
            process = Process(target=_run_code_in_subprocess, args=(code, result_queue))
            process.start()
            process.join(timeout=timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                return None, f"Code execution timed out after {timeout_seconds} seconds"
            if result_queue.empty():
                return None, "Code execution failed: no result returned"
            result_type, status, output = result_queue.get()
            if result_type == "error":
                return None, f"Code execution error: {status[:200]}"
            return status, output
        except Exception as exc:
            return None, f"Code execution error: {str(exc)[:200]}"

    async def judge_with_prompt(self, agent_name: str, prompt: str) -> bool:
        if not prompt or not str(prompt).strip():
            self._log_failure(agent_name, "judge_error", "empty judge prompt")
            return False

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    judgment = await self.judge_llm(prompt)
                    judgment = judgment.strip().upper()
                    judgment_start = judgment[:100]
                    if "INCORRECT" in judgment_start:
                        self._log_failure(agent_name, "judge_incorrect")
                        return False
                    if "CORRECT" in judgment_start:
                        return True
                    print(f"Warning: Unclear judgment for {agent_name}: {judgment[:200]}")
                    if "CORRECT" in judgment and "INCORRECT" not in judgment:
                        return True
                    self._log_failure(agent_name, "judge_incorrect", "ambiguous judge output")
                    return False
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                        await asyncio.sleep(1)
                    else:
                        raise
        except Exception as e:
            self._log_failure(agent_name, "judge_error", str(e))
            print(f"Error in judge evaluation for {agent_name}: {e}")
            return False

    def build_context(
        self,
        ground_truth: str,
        model_output: str,
        agent_name: str,
        sample_data: Optional[Dict[str, Any]] = None,
        problem: Optional[str] = None,
        solutions: Optional[List[Any]] = None,
        question: Optional[str] = None,
        workflow_type: Optional[str] = None,
        input_prompt: Optional[str] = None,
        original_sample: Optional[Dict[str, Any]] = None,
    ) -> JudgeContext:
        derived_problem = problem
        derived_solutions = solutions
        derived_question = question
        derived_original_sample = original_sample
        if isinstance(sample_data, dict):
            agent_input = sample_data.get("agent_input")
            if isinstance(agent_input, dict):
                if derived_problem is None:
                    candidate_problem = agent_input.get("problem")
                    if isinstance(candidate_problem, str):
                        derived_problem = candidate_problem
                if derived_solutions is None:
                    candidate_solutions = agent_input.get("solutions")
                    if isinstance(candidate_solutions, list):
                        derived_solutions = candidate_solutions
                if derived_question is None:
                    candidate_question = agent_input.get("problem")
                    if isinstance(candidate_question, str) and candidate_question.strip():
                        derived_question = candidate_question
            if derived_problem is None:
                candidate_problem = sample_data.get("problem")
                if isinstance(candidate_problem, str):
                    derived_problem = candidate_problem
            if derived_question is None:
                candidate_question = sample_data.get("problem")
                if isinstance(candidate_question, str) and candidate_question.strip():
                    derived_question = candidate_question
            if derived_original_sample is None:
                candidate_original_sample = sample_data.get("original_sample")
                if isinstance(candidate_original_sample, dict):
                    derived_original_sample = candidate_original_sample

        return JudgeContext(
            agent_name=agent_name,
            ground_truth=ground_truth,
            model_output=model_output,
            input_prompt=input_prompt,
            workflow_type=workflow_type or BenchmarkConfig.WORKFLOW_TYPE,
            sample_data=sample_data,
            problem=derived_problem,
            solutions=derived_solutions,
            question=derived_question,
            original_sample=derived_original_sample,
        )

    async def _evaluate_openclaw_context(self, context: JudgeContext) -> JudgeResult:
        validated = self._validate_openclaw_payloads(
            context.agent_name,
            context.model_output,
            context.ground_truth,
        )
        if validated is None:
            return JudgeResult(is_correct=False)

        predicted_payload, ground_truth_payload, required_fields = validated
        mode = str(self._openclaw_agent_policies()[context.agent_name].get("mode") or "").strip().lower()
        field_name = required_fields[0]
        if mode == "strict_exact":
            mismatches = []
            for required_field in required_fields:
                pred_norm = self._normalize_exact_value(predicted_payload[required_field])
                gt_norm = self._normalize_exact_value(ground_truth_payload[required_field])
                if pred_norm != gt_norm:
                    mismatches.append(
                        f"{required_field}: predicted='{pred_norm}', expected='{gt_norm}'"
                    )
            if mismatches:
                self._log_failure(
                    context.agent_name,
                    "strict_mismatch",
                    "; ".join(mismatches),
                )
                return JudgeResult(is_correct=False)
            return JudgeResult(is_correct=True)
        if mode != "semantic_llm":
            self._log_failure(
                context.agent_name,
                "judge_error",
                f"unsupported OpenClaw mode '{mode}'",
            )
            return JudgeResult(is_correct=False)

        prompt_template = str(
            self._openclaw_agent_policies()[context.agent_name].get("prompt") or ""
        ).strip()
        if not prompt_template:
            self._log_failure(context.agent_name, "judge_error", "missing semantic judge prompt")
            return JudgeResult(is_correct=False)

        if len(required_fields) == 1:
            predicted_value = str(predicted_payload[field_name]).strip()
            ground_truth_value = str(ground_truth_payload[field_name]).strip()
        else:
            predicted_value = json.dumps(
                {field: predicted_payload[field] for field in required_fields},
                ensure_ascii=False,
                sort_keys=True,
            )
            ground_truth_value = json.dumps(
                {field: ground_truth_payload[field] for field in required_fields},
                ensure_ascii=False,
                sort_keys=True,
            )
        prompt = prompt_template.format(
            input_prompt=context.input_prompt or "N/A",
            required_fields=", ".join(required_fields),
            ground_truth_field=ground_truth_value,
            predicted_field=predicted_value,
            ground_truth_json=ground_truth_value,
            predicted_json=predicted_value,
        )
        return JudgeResult(
            is_correct=await self.judge_with_prompt(context.agent_name, prompt)
        )

    async def evaluate_context(self, context: JudgeContext) -> JudgeResult:
        if not context.model_output or not context.model_output.strip():
            self._log_failure(context.agent_name, "missing_required_field", "model_output is empty")
            return JudgeResult(is_correct=False)

        if context.agent_name in self._openclaw_agent_policies():
            return await self._evaluate_openclaw_context(context)

        workflow_judge = self._get_workflow_judge(context.agent_name)
        if workflow_judge is None:
            self._log_failure(context.agent_name, "judge_error", "missing workflow judge")
            return JudgeResult(is_correct=False)
        try:
            return await workflow_judge(self, context)
        except Exception as exc:
            self._log_failure(context.agent_name, "judge_error", str(exc))
            print(f"Error in judge evaluation for {context.agent_name}: {exc}")
            traceback.print_exc()
            return JudgeResult(is_correct=False)

    async def evaluate(
        self,
        ground_truth: str,
        model_output: str,
        agent_name: str,
        problem: str = None,
        solutions: list = None,
        question: str = None,
        workflow_type: str = None,
        input_prompt: str = None,
        original_sample: Dict = None,
        sample_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        context = self.build_context(
            ground_truth=ground_truth,
            model_output=model_output,
            agent_name=agent_name,
            sample_data=sample_data,
            problem=problem,
            solutions=solutions,
            question=question,
            workflow_type=workflow_type,
            input_prompt=input_prompt,
            original_sample=original_sample,
        )
        result = await self.evaluate_context(context)
        return result.is_correct



class AgentBenchmarker:
    """Benchmarks a specific agent with different models and budgets"""
    
    def __init__(self, agent_name: str, judge: JudgeEvaluator):
        """
        Initialize benchmarker for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., "math_solver", "verifier")
            judge: Judge evaluator instance
        """
        self.agent_name = agent_name
        self.judge = judge
        self.llm_cache = {}  # Cache LLM instances

    async def aclose(self):
        """Close cached model clients."""
        for llm in list(self.llm_cache.values()):
            close_method = getattr(llm, "aclose", None)
            if close_method:
                try:
                    await close_method()
                except Exception:
                    pass
        self.llm_cache.clear()
    
    def _get_llm(self, model_name: str) -> AsyncLLM:
        """Get or create LLM instance (with caching)"""
        if model_name not in self.llm_cache:
            self.llm_cache[model_name] = create_llm_instance(model_name, endpoint_role="profile")
        return self.llm_cache[model_name]
    
    def _get_thinking_budgets(self, model_name: str) -> List[Any]:
        """Get appropriate thinking budgets for the model"""
        if "gpt-oss" in model_name.lower():
            return BenchmarkConfig.GPT_OSS_BUDGETS
        else:
            return BenchmarkConfig.SEARCH_BUDGETS
    
    async def _run_single_inference(
        self, 
        model_name: str, 
        thinking_budget: Any,
        input_prompt: str,
        ground_truth: str,
        sample_data: Dict = None
    ) -> Dict:
        """
        Run a single inference and evaluate it.
        
        Args:
            model_name: Name of the model to use
            thinking_budget: Thinking budget for the model
            input_prompt: Input prompt for the model
            ground_truth: Expected ground truth output
            sample_data: Full sample data (for extracting additional context)
        
        Returns:
            Dictionary with output, tokens, and correctness
        """
        llm = self._get_llm(model_name)

        try:
            output, input_tokens, output_tokens = await llm.call_with_thinking_budget(input_prompt, thinking_budget, return_io_tokens=True)
            context = self.judge.build_context(
                ground_truth=ground_truth,
                model_output=output,
                agent_name=self.agent_name,
                workflow_type=BenchmarkConfig.WORKFLOW_TYPE,
                input_prompt=input_prompt,
                sample_data=sample_data,
            )
            judge_result = await self.judge.evaluate_context(context)

            return {
                "output": output,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "is_correct": judge_result.is_correct,
                "metric_name": judge_result.metric_name,
                "metric_value": judge_result.metric_value,
                "error": False,
            }
        except Exception as e:
            print(f"Error during inference with model {model_name}, budget {thinking_budget}: {e}")
            return {
                "output": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "is_correct": False,
                "metric_name": None,
                "metric_value": None,
                "error": True,
            }


class BenchmarkRunner:
    """Main benchmark runner"""
    
    def __init__(self):
        """Initialize benchmark runner"""
        self.judge = JudgeEvaluator()
        self.training_data = None
        self.results = {}
        self.benchmarkers = []

    async def aclose(self):
        """Close all cached LLM clients used during profiling."""
        for benchmarker in list(self.benchmarkers):
            close_method = getattr(benchmarker, "aclose", None)
            if close_method:
                try:
                    await close_method()
                except Exception:
                    pass
        self.benchmarkers.clear()

        close_method = getattr(self.judge, "aclose", None)
        if close_method:
            try:
                await close_method()
            except Exception:
                pass
    
    def load_training_data(self) -> Dict[str, List[Dict]]:
        """
        Load and organize training data by agent.
        
        Returns:
            Dictionary mapping agent names to their samples
        """
        data_path = Path(BenchmarkConfig.TRAINING_DATA_PATH)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data structure
        if "training_data" not in data:
            raise ValueError("Training data file must contain 'training_data' key")
        
        # Check what agent names exist in the data
        available_agents = set(sample["agent_name"] for sample in data["training_data"])
        print(f"\nAvailable agent types in training data: {sorted(available_agents)}")
        print(f"Configured agent types to benchmark: {BenchmarkConfig.AGENT_NAMES}")

        filtered_agents = [agent for agent in BenchmarkConfig.AGENT_NAMES if agent in available_agents]
        removed_agents = [agent for agent in BenchmarkConfig.AGENT_NAMES if agent not in available_agents]
        if removed_agents:
            print(f"Skipping agents missing from data: {removed_agents}")
        BenchmarkConfig.AGENT_NAMES = filtered_agents
        if not BenchmarkConfig.AGENT_NAMES:
            raise ValueError("No configured profiling agents are present in training data")
        
        # Organize by agent
        agent_samples = defaultdict(list)
        for sample in data["training_data"]:
            agent_name = sample["agent_name"]
            if agent_name in BenchmarkConfig.AGENT_NAMES:
                # Validate required fields
                required_fields = ["raw_llm_prompt", "processed_output", "agent_name"]
                missing_fields = [f for f in required_fields if f not in sample]
                if missing_fields:
                    print(f"Warning: Sample missing fields {missing_fields}, skipping")
                    continue
                agent_samples[agent_name].append(sample)
        
        if BenchmarkConfig.MAX_SAMPLES_PER_AGENT is not None:
            cap = int(BenchmarkConfig.MAX_SAMPLES_PER_AGENT)
            if cap > 0:
                print(f"\nApplying sample cap per agent: {cap}")
                for agent_name in list(agent_samples.keys()):
                    agent_samples[agent_name] = agent_samples[agent_name][:cap]
        
        print("\nLoaded training data:")
        for agent_name, samples in agent_samples.items():
            print(f"  - {agent_name}: {len(samples)} samples")
        
        if not agent_samples:
            raise ValueError("No valid training samples found for configured agent types")
        
        self.training_data = dict(agent_samples)
        return self.training_data
    
    def _ensure_minimum_samples(self, agent_samples: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Ensure each agent has at least MIN_SAMPLES_PER_AGENT samples.
        If an agent has fewer samples, duplicate them until reaching the minimum.
        Duplicated samples are marked with '_is_duplicated' flag.
        
        Args:
            agent_samples: Dictionary mapping agent names to their samples
            
        Returns:
            Dictionary with potentially duplicated samples
        """
        if BenchmarkConfig.MIN_SAMPLES_PER_AGENT is None:
            return agent_samples
        
        min_samples = BenchmarkConfig.MIN_SAMPLES_PER_AGENT
        augmented_samples = {}
        
        for agent_name, samples in agent_samples.items():
            if len(samples) >= min_samples:
                augmented_samples[agent_name] = samples
                print(f"  - {agent_name}: {len(samples)} samples (meets minimum {min_samples})")
            else:
                # Duplicate samples to reach minimum
                original_count = len(samples)
                duplicated = []
                
                # Cycle through samples until we reach the minimum
                while len(duplicated) < min_samples:
                    for sample in samples:
                        if len(duplicated) >= min_samples:
                            break
                        # Create a copy and mark it as duplicated
                        dup_sample = sample.copy()
                        dup_sample["_is_duplicated"] = True
                        duplicated.append(dup_sample)
                
                augmented_samples[agent_name] = duplicated
                print(f"  - {agent_name}: {original_count} original → {len(duplicated)} total (duplicated to reach minimum {min_samples})")
        
        return augmented_samples
    
    async def run_benchmark(self):
        """Run the complete benchmark"""
        print("=" * 80)
        print("REASONING BUDGET BENCHMARK")
        print("=" * 80)
        
        # Load training data
        print("\nLoading training data...")
        agent_samples = self.load_training_data()
        
        # Ensure minimum samples per agent if configured
        if BenchmarkConfig.MIN_SAMPLES_PER_AGENT:
            print(f"\nEnsuring minimum {BenchmarkConfig.MIN_SAMPLES_PER_AGENT} samples per agent...")
            agent_samples = self._ensure_minimum_samples(agent_samples)
        
        # Create semaphore for global concurrency control
        semaphore = asyncio.Semaphore(BenchmarkConfig.MAX_CONCURRENT)
        
        # Build ALL tasks across ALL agents in one flat list
        all_tasks = []
        task_metadata = []
        
        for agent_name in BenchmarkConfig.AGENT_NAMES:
            if agent_name not in agent_samples:
                print(f"\nSkipping {agent_name}: no samples found")
                continue
            
            samples = agent_samples[agent_name]
            benchmarker = AgentBenchmarker(agent_name, self.judge)
            self.benchmarkers.append(benchmarker)
            
            for model_name in BenchmarkConfig.MODELS:
                budgets = benchmarker._get_thinking_budgets(model_name)
                
                for budget in budgets:
                    config_key = f"{model_name}_budget_{budget}"
                    
                    for sample_idx, sample in enumerate(samples):
                        if sample["agent_name"] == "sc_ensemble":
                            ground_truth = sample["raw_llm_output"]
                        else:
                            ground_truth = sample["processed_output"]
                        input_prompt = sample["raw_llm_prompt"]
                        
                        # Determine number of repetitions: use 1 if sample is duplicated, else use N_REPETITIONS
                        is_duplicated = sample.get("_is_duplicated", False)
                        n_repetitions = 1 if is_duplicated else BenchmarkConfig.N_REPETITIONS
                        
                        # Create N repetition tasks for this sample
                        for rep_idx in range(n_repetitions):
                            async def bounded_task(
                                b=benchmarker, 
                                model=model_name, 
                                bud=budget, 
                                prompt=input_prompt, 
                                gt=ground_truth, 
                                s_data=sample
                            ):
                                async with semaphore:
                                    return await b._run_single_inference(model, bud, prompt, gt, s_data)
                            
                            all_tasks.append(bounded_task())
                            task_metadata.append({
                                "agent_name": agent_name,
                                "config_key": config_key,
                                "model": model_name,
                                "budget": budget,
                                "sample_idx": sample_idx,
                                "sample": sample,
                                "rep_idx": rep_idx,
                                "is_duplicated": is_duplicated
                            })
        
        # Shuffle tasks for balanced GPU load
        paired_tasks = list(zip(all_tasks, task_metadata))
        random.shuffle(paired_tasks)
        all_tasks, task_metadata = zip(*paired_tasks)
        all_tasks = list(all_tasks)
        task_metadata = list(task_metadata)
        
        # Run ALL tasks in parallel with single progress bar
        total_tasks = len(all_tasks)
        total_agents = len([a for a in BenchmarkConfig.AGENT_NAMES if a in agent_samples])
        if BenchmarkConfig.DEBUG:
            print(f"\nRunning {total_tasks} total inference tasks in SEQUENTIAL DEBUG mode...")
        else:
            print(f"\nRunning {total_tasks} total inference tasks in FULLY PARALLEL mode...")
        print(f"  - Agents: {total_agents}")
        print(f"  - Models: {len(BenchmarkConfig.MODELS)}")
        print(f"  - Max concurrent: {BenchmarkConfig.MAX_CONCURRENT}")
        print(f"  - Tasks shuffled for balanced GPU load: {'No (Debug mode)' if BenchmarkConfig.DEBUG else 'Yes'}")
        
        results = await tqdm.gather(*all_tasks, desc="All agents | All configs")
        
        # Organize results by agent and configuration
        organized_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for task_result, metadata in zip(results, task_metadata):
            agent_name = metadata["agent_name"]
            config_key = metadata["config_key"]
            sample_idx = metadata["sample_idx"]
            organized_results[agent_name][config_key][sample_idx].append(task_result)
        
        # Aggregate results for each agent, configuration, and sample
        all_results = {}
        
        for agent_name in organized_results:
            samples = agent_samples[agent_name]
            all_results[agent_name] = {}
            
            for config_key in organized_results[agent_name]:
                config_results = []
                
                # Extract model and budget from config_key
                parts = config_key.split("_budget_")
                model_name = parts[0]
                budget = parts[1]
                
                for sample_idx in sorted(organized_results[agent_name][config_key].keys()):
                    sample = samples[sample_idx]
                    repetition_results = organized_results[agent_name][config_key][sample_idx]
                    repetition_results = [r for r in repetition_results if not r.get("error", False)]
                    
                    # Aggregate across repetitions
                    n_reps = len(repetition_results)
                    accuracy = sum(r["is_correct"] for r in repetition_results) / n_reps if n_reps > 0 else 0
                    avg_input_tokens = sum(r["input_tokens"] for r in repetition_results) / n_reps if n_reps > 0 else 0
                    avg_output_tokens = sum(r["output_tokens"] for r in repetition_results) / n_reps if n_reps > 0 else 0

                    metric_names = {
                        r.get("metric_name")
                        for r in repetition_results
                        if r.get("metric_name")
                    }
                    metric_name = metric_names.pop() if len(metric_names) == 1 else None
                    avg_metric_value = None
                    if metric_name:
                        metric_values = [
                            r.get("metric_value", 0.0)
                            for r in repetition_results
                            if r.get("metric_name") == metric_name and r.get("metric_value") is not None
                        ]
                        if metric_values:
                            avg_metric_value = sum(metric_values) / len(metric_values)
                    
                    result_entry = {
                        "agent_name": agent_name,
                        "model": model_name,
                        "thinking_budget": budget,
                        "problem": sample.get("problem", "N/A"),
                        "sample_timestamp": sample.get("sample_timestamp", "N/A"),
                        "step_number": sample.get("step_number", "N/A"),
                        "n_repetitions": n_reps,
                        "accuracy": accuracy,
                        "avg_input_tokens": avg_input_tokens,
                        "avg_output_tokens": avg_output_tokens,
                        "all_runs": repetition_results
                    }
                    
                    if metric_name:
                        result_entry["metric_name"] = metric_name
                    if avg_metric_value is not None and metric_name:
                        result_entry[f"avg_{metric_name}"] = avg_metric_value
                    
                    config_results.append(result_entry)
                
                all_results[agent_name][config_key] = config_results
        
        self.results = all_results
        return all_results
    
    def compute_summary_statistics(self) -> Dict:
        """Compute summary statistics across all configurations"""
        summary = {}
        
        for agent_name, agent_configs in self.results.items():
            summary[agent_name] = {}
            
            for config_key, config_results in agent_configs.items():
                # Extract model and budget from config_key
                # Format: "model_name_budget_value"
                parts = config_key.split("_budget_")
                model_name = parts[0]
                budget = parts[1] if len(parts) > 1 else "unknown"
                
                # Compute statistics
                n_samples = len(config_results)
                overall_accuracy = sum(r["accuracy"] for r in config_results) / n_samples if n_samples > 0 else 0
                avg_input_tokens = sum(r["avg_input_tokens"] for r in config_results) / n_samples if n_samples > 0 else 0
                avg_output_tokens = sum(r["avg_output_tokens"] for r in config_results) / n_samples if n_samples > 0 else 0
                total_tokens = avg_input_tokens + avg_output_tokens
                
                summary_entry = {
                    "model": model_name,
                    "thinking_budget": budget,
                    "n_samples": n_samples,
                    "overall_accuracy": overall_accuracy,
                    "avg_input_tokens_per_sample": avg_input_tokens,
                    "avg_output_tokens_per_sample": avg_output_tokens,
                    "avg_total_tokens_per_sample": total_tokens,
                    "accuracy_percent": f"{overall_accuracy * 100:.2f}%"
                }
                
                metric_names = {
                    r.get("metric_name")
                    for r in config_results
                    if r.get("metric_name")
                }
                metric_name = metric_names.pop() if len(metric_names) == 1 else None
                if metric_name:
                    metric_key = f"avg_{metric_name}"
                    metric_values = [r.get(metric_key, 0.0) for r in config_results if metric_key in r]
                    if metric_values:
                        overall_metric = sum(metric_values) / len(metric_values)
                        summary_entry[f"overall_{metric_name}"] = overall_metric
                        summary_entry[f"{metric_name}_percent"] = f"{overall_metric * 100:.2f}%"
                
                summary[agent_name][config_key] = summary_entry
        
        return summary
    
    def save_results(self, output_dir: Optional[Path] = None):
        """Save results to JSON files"""
        if output_dir is None:
            output_dir = Path(BenchmarkConfig.OUTPUT_DIR)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / f"benchmark_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to: {output_dir}")
        
        # Save detailed results
        detailed_path = output_dir / "detailed_results.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"  - Saved detailed results: {detailed_path}")
        
        # Save summary statistics
        summary = self.compute_summary_statistics()
        summary_path = output_dir / "summary_statistics.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  - Saved summary statistics: {summary_path}")
        
        # Save configuration info
        config_info = {
            "benchmark_config": {
                "models": BenchmarkConfig.MODELS,
                "search_budgets": BenchmarkConfig.SEARCH_BUDGETS,
                "gpt_oss_budgets": BenchmarkConfig.GPT_OSS_BUDGETS,
                "judge_model": BenchmarkConfig.JUDGE_MODEL,
                "n_repetitions": BenchmarkConfig.N_REPETITIONS,
                "agent_names": BenchmarkConfig.AGENT_NAMES,
                "training_data_path": BenchmarkConfig.TRAINING_DATA_PATH
            },
            "run_info": {
                "timestamp": timestamp,
                "output_directory": str(output_dir)
            }
        }
        config_path = output_dir / "benchmark_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        print(f"  - Saved configuration: {config_path}")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        for agent_name, configs in summary.items():
            print(f"\n{agent_name.upper()}:")
            print("-" * 80)
            print(f"{'Model':<30} {'Budget':<15} {'Accuracy':<12} {'Avg Tokens':<15}")
            print("-" * 80)
            
            for config_key, stats in configs.items():
                model = stats['model']
                budget = stats['thinking_budget']
                accuracy = stats['accuracy_percent']
                tokens = f"{stats['avg_total_tokens_per_sample']:.1f}"
                
                print(f"{model:<30} {budget:<15} {accuracy:<12} {tokens:<15}")
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {output_dir}")
        print("=" * 80)
        
        return output_dir


async def run_profiling(
    experiment_id: str,
    models: Optional[List[str]] = None,
    search_budgets: Optional[List[Any]] = None,
    max_samples: Optional[int] = None,
    max_concurrent: int = 128,
    debug: bool = False,
    min_samples_per_agent: Optional[int] = 100,
    livecodebench_validate_file: Optional[str] = None,
    livecodebench_public_test_file: Optional[str] = None,
    workflow_type: Optional[str] = None,
    training_data_path: Optional[str] = None,
    experiment_root: Optional[str] = None,
    openclaw_lobster_workflow_file: Optional[str] = None,
    openclaw_agent_policies: Optional[Dict[str, Any]] = None,
    judge_policies: Optional[Any] = None,
) -> Path:
    """Run reasoning-budget profiling for an experiment."""
    BenchmarkConfig.initialize_from_experiment_id(
        experiment_id,
        search_budgets=search_budgets,
        workflow_type=workflow_type,
        training_data_path=training_data_path,
        experiment_root=experiment_root,
        openclaw_lobster_workflow_file=openclaw_lobster_workflow_file,
        openclaw_agent_policies=openclaw_agent_policies,
        judge_policies=judge_policies,
    )

    if models:
        BenchmarkConfig.MODELS = models
        print(f"Override: Models = {BenchmarkConfig.MODELS}")

    if max_concurrent:
        BenchmarkConfig.MAX_CONCURRENT = max_concurrent
        print(f"Override: Max concurrent = {BenchmarkConfig.MAX_CONCURRENT}")
    
    if max_samples is not None:
        BenchmarkConfig.MAX_SAMPLES_PER_AGENT = max_samples
        print(f"Override: Max samples per agent = {BenchmarkConfig.MAX_SAMPLES_PER_AGENT}")

    if debug:
        BenchmarkConfig.DEBUG = True
        BenchmarkConfig.MAX_CONCURRENT = 1
        print("Override: DEBUG mode ENABLED - Running in SEQUENTIAL mode (MAX_CONCURRENT = 1)")

    if min_samples_per_agent:
        BenchmarkConfig.MIN_SAMPLES_PER_AGENT = min_samples_per_agent
        print(f"Override: MIN_SAMPLES_PER_AGENT = {BenchmarkConfig.MIN_SAMPLES_PER_AGENT}")

    if livecodebench_validate_file:
        BenchmarkConfig.LIVECODEBENCH_VALIDATE_PATH = str(livecodebench_validate_file)
        print(f"Override: LIVECODEBENCH_VALIDATE_PATH = {BenchmarkConfig.LIVECODEBENCH_VALIDATE_PATH}")

    if livecodebench_public_test_file:
        BenchmarkConfig.LIVECODEBENCH_PUBLIC_TEST_PATH = str(livecodebench_public_test_file)
        print(f"Override: LIVECODEBENCH_PUBLIC_TEST_PATH = {BenchmarkConfig.LIVECODEBENCH_PUBLIC_TEST_PATH}")

    runner = BenchmarkRunner()
    try:
        await runner.run_benchmark()
        output_dir = runner.save_results()
        print(f"\nBenchmark complete! Results saved to: {output_dir}")
        return output_dir
    finally:
        close_method = getattr(runner, "aclose", None)
        if close_method:
            await close_method()
