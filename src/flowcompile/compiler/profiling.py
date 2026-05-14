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
import concurrent.futures
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

from flowcompile.core.llm.client import AsyncLLM, create_llm_instance
from flowcompile.core.workflow.operators import run_code
from flowcompile.core.llm.formatter import XmlFormatter
from flowcompile.core.workflow.operators import ScEnsembleOp, AnswerGenerateOp
from flowcompile.benchmarks.livecodebench import evaluate_generations_by_problem
from flowcompile.core.data_paths import resolve_existing_path
from concurrent.futures import ProcessPoolExecutor
from flowcompile.workflows.dsl_registry import get_workflow_module

GLOBAL_DEFAULT_SEARCH_BUDGETS = (10, 200, 400, 800, 1000, 1500, 2000, 3000, 4000, 5000)


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
    results_dir = Path("results") / experiment_id
    profile_dir = results_dir / "01_profile"
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {results_dir}")
    
    # Look for aggregated training data in canonical and legacy locations first.
    training_data_file = None

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
    
    # Determine workflow type from experiment ID or training data
    # Note: 'math', 'math500', and 'gsm8k' datasets all use workflow_type='math'
    # since they share the same workflow structure.
    if "math" in experiment_id.lower() or "gsm8k" in experiment_id.lower():
        workflow_type = "math"
    elif "hotpotqa" in experiment_id.lower():
        workflow_type = "hotpotqa"
    elif "livecodebench" in experiment_id.lower():
        workflow_type = "livecodebench"
    else:
        # Default fallback
        workflow_type = "math"
    
    workflow_module = get_workflow_module(workflow_type)
    inferred_agents = workflow_module.infer_profiling_agents()
    selected_search_budgets = list(search_budgets) if search_budgets else list(GLOBAL_DEFAULT_SEARCH_BUDGETS)

    return {
        "training_data_path": training_data_file,
        "output_dir": str(profile_dir),
        "workflow_type": workflow_type,
        "search_budgets": selected_search_budgets,
        "agent_names": inferred_agents,
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
    LIVECODEBENCH_VALIDATE_PATH = "data/livecodebench_validate.jsonl"
    LIVECODEBENCH_PUBLIC_TEST_PATH = "data/livecodebench_public_test.jsonl"
    
    @classmethod
    def initialize_from_experiment_id(
        cls,
        experiment_id: str,
        search_budgets: Optional[List[Any]] = None,
    ):
        """
        Initialize configuration from experiment ID.
        
        Args:
            experiment_id: Experiment ID (e.g., "1208_math500", "1221_hotpotqa")
        """
        config = get_experiment_config(experiment_id, search_budgets=search_budgets)
        cls.TRAINING_DATA_PATH = config["training_data_path"]
        cls.OUTPUT_DIR = config["output_dir"]
        cls.SEARCH_BUDGETS = config["search_budgets"]
        cls.AGENT_NAMES = config["agent_names"]
        cls.WORKFLOW_TYPE = config["workflow_type"]
        
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

    MATH_JUDGE_PROMPT = """You are evaluating a mathematical solution.

Ground Truth:
{ground_truth}

Model Output:
{model_output}

Task: Compare the final answers (e.g., \\boxed{{...}}) from the model output to the ground truth. Ignore minor formatting differences.

Respond with ONLY ONE WORD:
- "CORRECT" if final answers match
- "INCORRECT" if they differ

Judgment:"""
    
    HOTPOTQA_JUDGE_PROMPT = """You are evaluating a question answering output.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Model Output:
{model_output}

Task: Compare the answer in the model output with the ground truth. Consider semantic equivalence - answers can be phrased differently but mean the same thing. Ignore minor formatting differences.

Respond with ONLY ONE WORD:
- "CORRECT" if answers are semantically equivalent
- "INCORRECT" if they differ in meaning

Judgment:"""
    
    # Task-specific judge prompts for different agent types
    JUDGE_PROMPTS = {
        "programmer": """You are evaluating whether a Python code's execution output is correct.

Expected Output (from ground truth):
{ground_truth}

Actual Execution Output:
{exec_output}

Task: Compare the actual output with the expected output. Consider numerical equivalence (e.g., 0.0 = 0, 1.5 = 1.50) and ignore minor formatting differences (whitespace, trailing zeros).

Respond with ONLY ONE WORD:
- "CORRECT" if outputs match
- "INCORRECT" if outputs differ

Judgment:""",
        "detailed_solver": MATH_JUDGE_PROMPT,
        "generate_solver": MATH_JUDGE_PROMPT,
        "refine_solver": MATH_JUDGE_PROMPT,
        "answer_generate": HOTPOTQA_JUDGE_PROMPT,
        "sc_ensemble_math": """You are evaluating an ensemble agent's solution selection for a MATH problem.

Ground Truth Solution:
{ground_truth_solution}

Predicted Solution:
{predicted_solution}

Ground Truth Reasoning:
{ground_truth_output}

Agent's Reasoning:
{model_output}

Task: 
1. First, compare the final mathematical answers from both solutions. Are they equivalent?
2. Then, check if the agent's reasoning process is correct and aligns with the ground truth reasoning.

Respond with ONLY ONE WORD:
- "CORRECT" if the predicted solution's final answer matches the ground truth solution's final answer AND the reasoning process is correct
- "INCORRECT" otherwise

Judgment:""",
        "sc_ensemble_hotpotqa": """You are evaluating an ensemble agent's answer selection for a question answering task.

Question:
{question}

Ground Truth Answer:
{ground_truth_solution}

Predicted Answer:
{predicted_solution}

Ground Truth Reasoning:
{ground_truth_output}

Agent's Reasoning:
{model_output}

Task: 
1. Compare the ground truth answer with the predicted answer. Are they semantically equivalent?
2. Then, check if the agent's reasoning process is correct and aligns with the ground truth reasoning.

Respond with ONLY ONE WORD:
- "CORRECT" if the predicted answer matches the ground truth answer semantically AND the reasoning process is correct
- "INCORRECT" otherwise

Judgment:""",
        "sc_ensemble_livecodebench": """You are evaluating an ensemble agent's code solution selection for a coding problem.

Problem:
{problem}

Ground Truth Solution:
{ground_truth_solution}

Predicted Solution:
{predicted_solution}

Ground Truth Reasoning:
{ground_truth_output}

Agent's Reasoning:
{model_output}

Task: 
1. First, compare the code solutions. Are they functionally equivalent (same logic and output)?
2. Then, check if the agent's reasoning process for selecting the solution is correct and aligns with the ground truth reasoning.

Respond with ONLY ONE WORD:
- "CORRECT" if the predicted solution is functionally equivalent to the ground truth solution AND the reasoning process is correct
- "INCORRECT" otherwise

Judgment:""",
    }
    
    def __init__(self, judge_model: str = BenchmarkConfig.JUDGE_MODEL):
        """Initialize with judge model"""
        self.judge_llm = create_llm_instance(judge_model)
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

    def _get_prompt_template(self, agent_name: str) -> str:
        """Get the appropriate prompt template for the given agent"""
        return self.JUDGE_PROMPTS[agent_name]
    
    def _normalize_answer(self, s: str) -> str:
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
    def _extract_boxed_choice_letter(text: str) -> Optional[str]:
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
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
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

    def _extract_code_from_output(self, output: str) -> Optional[str]:
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
    
    async def evaluate(self, ground_truth: str, model_output: str, agent_name: str, 
                      problem: str = None, solutions: list = None, question: str = None,
                      workflow_type: str = None, input_prompt: str = None, 
                      original_sample: Dict = None) -> bool:
        """
        Evaluate if model output matches ground truth.
        
        Args:
            ground_truth: The expected/ground truth output
            model_output: The output from the model to evaluate
            agent_name: The name of the agent (for task-specific evaluation)
            problem: The original problem statement (for sc_ensemble or HotpotQA)
            solutions: List of candidate solutions (for sc_ensemble)
            question: The question being answered (for HotpotQA agents)
            workflow_type: The workflow type (e.g., 'math', 'hotpotqa') for sc_ensemble
            original_sample: Original sample data (for private test evaluation)
        
        Returns:
            True if correct, False otherwise
        """
        assert agent_name != "format_answer", "Use evaluate_with_f1 for format_answer agent"
        # Handle empty outputs
        if not model_output or not model_output.strip():
            return False
        
        # Special handling for code_generate and reflection_test agents - use private test evaluation
        if agent_name in ["code_generate", "reflection_test"] and workflow_type == "livecodebench":
            try:
                model_code = self._extract_code_from_output(model_output)
                if model_code is None:
                    model_code = model_output  # Fallback to raw output if extraction fails

                if not model_code or not model_code.strip():
                    print(f"Warning: Empty code output for {agent_name}")
                    return False
                
                # Evaluate using private test cases from original_sample
                if original_sample is None:
                    print(f"Warning: No original_sample provided for private test evaluation of {agent_name}")
                    return False
                
                # Use the private test evaluator
                return await self.evaluate_code_with_private_tests(model_code, original_sample)
                
            except Exception as e:
                print(f"Error in code evaluation for {agent_name}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Special handling for programmer agent - execute code and check result with run_code
        elif agent_name == "programmer":
            try:
                # Extract code from model output
                model_code = self._extract_code_from_output(model_output)
                
                if not model_code:
                    print(f"Warning: Could not extract code from model output")
                    return False
                
                # Execute the model's generated code with timeout
                # Use multiprocessing to isolate code execution in a separate process
                # This prevents blocking/infinite loops from affecting the main process
                try:
                    from multiprocessing import Process, Queue

                    result_queue = Queue()
                    process = Process(target=_run_code_in_subprocess, args=(model_code, result_queue))
                    process.start()
                    
                    # Wait for process to complete with timeout
                    process.join(timeout=5.0)
                    
                    if process.is_alive():
                        # Process is still running after timeout - force kill it
                        process.terminate()
                        process.join(timeout=1.0)
                        if process.is_alive():
                            process.kill()
                            process.join()
                        print(f"Code execution timed out after 5 seconds")
                        return False
                    
                    # Get result from queue
                    if not result_queue.empty():
                        result_type, status, output = result_queue.get()
                        if result_type == "error":
                            print(f"Code execution error: {status[:200]}")
                            return False
                    else:
                        print(f"Code execution failed: no result returned")
                        return False
                        
                except Exception as e:
                    print(f"Code execution error: {str(e)[:200]}")
                    return False
                
                # If execution failed, it's incorrect
                if status != "Success":
                    print(f"Code execution failed: {output[:200]}")
                    return False
                
                # Use LLM judge to compare execution output with ground truth
                # Pass the raw ground truth text directly to the prompt
                prompt_template = self._get_prompt_template(agent_name)
                prompt = prompt_template.format(
                    ground_truth=ground_truth,
                    exec_status=status,
                    exec_output=output
                )
                            
            except Exception as e:
                print(f"Error in programmer evaluation: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            # Get task-specific prompt for other agents
            prompt_template = self._get_prompt_template(agent_name)
            
            # For answer_generate, validate XML format and extract answer field
            if agent_name == "answer_generate":
                # First, validate the response format using XmlFormatter
                formatter = XmlFormatter.from_model(AnswerGenerateOp)
                is_valid_format, parsed_data = formatter.validate_response(model_output)
                
                # Check if format is valid and contains answer field
                if not is_valid_format or not parsed_data:
                    return False
                
                if "answer" not in parsed_data or not parsed_data["answer"].strip():
                    return False

                
                # Extract the answer from model output
                model_answer = parsed_data["answer"].strip()
                
                # Ground truth is already the processed answer (not XML format)
                ground_truth_answer = ground_truth.strip()
                
                # Extract question from problem if it's a formatted string
                if question is None and problem:
                    if "Question:" in problem:
                        question = problem.split("Question:")[-1].split("Answer:")[0].strip()
                    else:
                        question = problem
                
                prompt = prompt_template.format(
                    ground_truth=ground_truth_answer,
                    model_output=model_answer,
                    question=question or "N/A"
                )
            # For sc_ensemble, we need additional context and format validation
            elif agent_name.startswith("sc_ensemble"):
                formatter = XmlFormatter.from_model(ScEnsembleOp)
                predicted_solution_letter = self._extract_boxed_choice_letter(model_output)
                if not predicted_solution_letter:
                    return False

                # Try structured parse first, then regex fallback.
                ground_truth_solution_letter = None
                try:
                    _ok, ground_truth_parsed_data = formatter.validate_response(ground_truth)
                    if isinstance(ground_truth_parsed_data, dict):
                        candidate = ground_truth_parsed_data.get("solution_letter", "")
                        if isinstance(candidate, str) and candidate.strip():
                            ground_truth_solution_letter = candidate.strip().upper()
                except Exception:
                    ground_truth_solution_letter = None

                if not ground_truth_solution_letter:
                    ground_truth_solution_letter = self._extract_boxed_choice_letter(ground_truth)
                if not ground_truth_solution_letter:
                    return False

                # If trace data doesn't carry candidate solutions (common in DSL traces),
                # fall back to direct choice-letter agreement to avoid dropping samples.
                if not isinstance(solutions, list) or len(solutions) == 0:
                    return predicted_solution_letter == ground_truth_solution_letter

                gt_idx = ord(ground_truth_solution_letter) - ord('A')
                pred_idx = ord(predicted_solution_letter) - ord('A')
                if gt_idx < 0 or pred_idx < 0 or gt_idx >= len(solutions) or pred_idx >= len(solutions):
                    return predicted_solution_letter == ground_truth_solution_letter

                ground_truth_solution = solutions[gt_idx]
                try:
                    predicted_solution = solutions[pred_idx]
                except Exception:
                    return predicted_solution_letter == ground_truth_solution_letter
                
                # Use workflow_type from parameter (passed from config) or fall back to config
                if workflow_type is None:
                    workflow_type = BenchmarkConfig.WORKFLOW_TYPE
                
                is_hotpotqa = workflow_type == "hotpotqa"
                is_livecodebench = workflow_type == "livecodebench"
                
                if is_hotpotqa:
                    # HotpotQA: extract question and use HotpotQA prompt
                    if question is None and problem:
                        if "Question:" in problem:
                            question = problem.split("Question:")[-1].split("Answer:")[0].strip()
                        else:
                            question = problem
                    
                    prompt_template = self.JUDGE_PROMPTS["sc_ensemble_hotpotqa"]
                    prompt = prompt_template.format(
                        question=question or "N/A",
                        ground_truth_solution=ground_truth_solution, 
                        predicted_solution=predicted_solution, 
                        model_output=model_output, 
                        ground_truth_output=ground_truth
                    )
                elif is_livecodebench:
                    # LiveCodeBench: use code evaluation prompt
                    prompt_template = self.JUDGE_PROMPTS["sc_ensemble_livecodebench"]
                    prompt = prompt_template.format(
                        problem=problem or "N/A",
                        ground_truth_solution=ground_truth_solution, 
                        predicted_solution=predicted_solution, 
                        model_output=model_output, 
                        ground_truth_output=ground_truth
                    )
                else:
                    # MATH: use MATH prompt
                    prompt_template = self.JUDGE_PROMPTS["sc_ensemble_math"]
                    prompt = prompt_template.format(
                        ground_truth_solution=ground_truth_solution, 
                        predicted_solution=predicted_solution, 
                        model_output=model_output, 
                        ground_truth_output=ground_truth
                    )
            else:
                prompt = prompt_template.format(
                    ground_truth=ground_truth,
                    model_output=model_output
                )
        
        # Common judgment logic for all agents
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    judgment = await self.judge_llm(prompt)
                    judgment = judgment.strip().upper()

                    # Parse judgment - look for clear indicators
                    # Must find "CORRECT" and NOT find "INCORRECT" 
                    # Check first 100 chars for the judgment word
                    judgment_start = judgment[:100]
                    
                    if "INCORRECT" in judgment_start:
                        return False
                    elif "CORRECT" in judgment_start:
                        return True
                    else:
                        # Unclear judgment, try to parse more carefully
                        print(f"Warning: Unclear judgment for {agent_name}: {judgment[:200]}")
                        # Default to checking the full judgment
                        if "CORRECT" in judgment and "INCORRECT" not in judgment:
                            return True
                        else:
                            return False
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                        await asyncio.sleep(1)
                    else:
                        raise
            
        except Exception as e:
            print(f"Error in judge evaluation for {agent_name}: {e}")
            return False



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
            self.llm_cache[model_name] = create_llm_instance(model_name)
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
            # NOTE: already handled in operators.py
            # if self.agent_name == "sc_ensemble":
            #     input_prompt = ensemble_prompt_surgery(input_prompt)
            # Call with thinking budget
            output, input_tokens, output_tokens = await llm.call_with_thinking_budget(input_prompt, thinking_budget, return_io_tokens=True)

            # Evaluate correctness with agent-specific evaluation
            # For format_answer, use F1 metric
            if self.agent_name == "format_answer" and sample_data:
                is_correct, f1_score = await self.judge.evaluate_with_f1(
                    ground_truth, output, 
                    agent_name=self.agent_name
                )
                return {
                    "output": output,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "is_correct": is_correct,
                    "f1_score": f1_score,  # Store F1 score separately
                    "error": False,
                }
            # For sc_ensemble, pass additional context and use configured workflow type
            elif self.agent_name == "sc_ensemble" and sample_data:
                problem = sample_data.get("agent_input", {}).get("problem", None)
                solutions = sample_data.get("agent_input", {}).get("solutions", None)
                
                # Use configured workflow type
                workflow_type = BenchmarkConfig.WORKFLOW_TYPE
                
                # Determine the specific agent name variant based on workflow type
                agent_name_variant = f"sc_ensemble_{workflow_type}"
                
                # Extract question for HotpotQA
                question = None
                if workflow_type == "hotpotqa":
                    full_problem = sample_data.get("problem", "")
                    if "Question:" in full_problem:
                        question = full_problem.split("Question:")[-1].split("Answer:")[0].strip()
                    else:
                        question = problem
                
                # Get original_sample for private test evaluation
                original_sample = sample_data.get("original_sample", None)
                
                is_correct = await self.judge.evaluate(
                    ground_truth, output, 
                    agent_name=agent_name_variant,
                    problem=problem,
                    solutions=solutions,
                    question=question,
                    workflow_type=workflow_type,
                    input_prompt=input_prompt,
                    original_sample=original_sample
                )
            # For HotpotQA agents, pass question context
            elif self.agent_name == "answer_generate" and sample_data:
                problem = sample_data.get("problem", "")
                agent_input = sample_data.get("agent_input", {})
                question = agent_input.get("problem", problem) if isinstance(agent_input, dict) else problem
                original_sample = sample_data.get("original_sample", None)
                
                is_correct = await self.judge.evaluate(
                    ground_truth, output, 
                    agent_name=self.agent_name,
                    problem=problem,
                    question=question,
                    original_sample=original_sample
                )
            else:
                # Get original_sample and workflow_type for all other agents
                original_sample = sample_data.get("original_sample", None) if sample_data else None
                workflow_type = BenchmarkConfig.WORKFLOW_TYPE
                
                is_correct = await self.judge.evaluate(
                    ground_truth, output, 
                    agent_name=self.agent_name,
                    workflow_type=workflow_type,
                    original_sample=original_sample
                )

            return {
                "output": output,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "is_correct": is_correct,
                "f1_score": None,  # Only format_answer has F1 score
                "error": False,
            }
        except Exception as e:
            print(f"Error during inference with model {model_name}, budget {thinking_budget}: {e}")
            return {
                "output": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "is_correct": False,
                "f1_score": None,
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
                    
                    # For format_answer, also compute average F1 score
                    avg_f1_score = None
                    if agent_name == "format_answer":
                        f1_scores = [r.get("f1_score", 0.0) for r in repetition_results if r.get("f1_score") is not None]
                        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                    
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
                    
                    if avg_f1_score is not None:
                        result_entry["avg_f1_score"] = avg_f1_score
                    
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
                
                # For format_answer, add average F1 score
                if agent_name == "format_answer":
                    f1_scores = [r.get("avg_f1_score", 0.0) for r in config_results if "avg_f1_score" in r]
                    if f1_scores:
                        overall_f1 = sum(f1_scores) / len(f1_scores)
                        summary_entry["overall_f1_score"] = overall_f1
                        summary_entry["f1_score_percent"] = f"{overall_f1 * 100:.2f}%"
                
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
) -> Path:
    """Run reasoning-budget profiling for an experiment."""
    BenchmarkConfig.initialize_from_experiment_id(
        experiment_id,
        search_budgets=search_budgets,
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
