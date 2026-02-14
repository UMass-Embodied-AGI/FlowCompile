import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from workflow_compiler.benchmarks.benchmark import BaseBenchmark
from workflow_compiler.benchmarks.registry import register_benchmark
from workflow_compiler.core.llm.client import AsyncLLM
from workflow_compiler.core.logs import logger
from copy import deepcopy


@register_benchmark()
class MATHBenchmark(BaseBenchmark):
    BENCHMARK_NAME = "MATH"
    ALIASES = ["math", "MATH", "math500", "MATH500"]
    WORKFLOW_TYPE = "math"
    METRIC_NAME = "accuracy"
    DEFAULT_SPLIT_PATHS = {
        "validate": "data/math_validate.jsonl",
        "test": "data/math_test.jsonl",
    }

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.llm = AsyncLLM('gpt-5-mini')
        self._score_cache = {}  # Cache for calculate_score results

    def extract_model_answer(self, text: str) -> str:
        text = text or ""
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    async def judge_correctness(self, problem, final_answer, ground_truth):
        prompt = f"""Determine if the following answer is correct for the given problem.

Problem: {problem}

Answer: {final_answer}

Ground Truth: {ground_truth}

Is the answer correct? Respond with only 'yes' or 'no'."""
        response = await self.llm(prompt)
        return response.strip().lower() == 'yes'

    async def calculate_score(self, problem: dict | str, expected_output: str, prediction: str) -> Tuple[int, str]:
        if isinstance(problem, dict):
            problem_text = problem.get("problem", problem)
        else:
            problem_text = problem
        
        # Create cache key
        cache_key = (problem_text, expected_output, prediction)
        
        # Check cache
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        
        predicted_answer = self.extract_model_answer(prediction)
        
        # Fix: If prediction or extracted answer is empty, return wrong answer directly
        # This prevents the LLM judge from incorrectly marking empty answers as correct
        if not prediction or not prediction.strip() or not predicted_answer or not predicted_answer.strip():
            result = (0, predicted_answer)
            self._score_cache[cache_key] = result
            return result
        
        is_correct = await self.judge_correctness(problem_text, prediction, expected_output)
        
        result = (1 if is_correct else 0, predicted_answer)
        
        # Cache the result
        self._score_cache[cache_key] = result
        
        return result


    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int]:
        input_text = problem["problem"]
        expected_output = problem["answer"] if "answer" in problem else problem["solution"]
        try:
            # Pass the full problem dict to the workflow so it can extract ground truth
            output = await self._generate_output(graph, problem)
            uni_score, extracted_output = await self.calculate_score(problem, expected_output, output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )

            return input_text, output, expected_output, uni_score

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score"]

    @classmethod
    def score_from_result(cls, result: Tuple[Any, ...]) -> float:
        return float(result[-1])

    @classmethod
    def result_key(cls, result: Tuple[Any, ...]) -> str:
        return str(result[0]) if result else ""

    @classmethod
    def trace_key(cls, trace: dict) -> str:
        orig = trace.get("metadata", {}).get("original_sample", {}) or {}
        problem = orig.get("problem") or orig.get("question") or trace.get("problem")
        return str(problem) if problem is not None else ""
