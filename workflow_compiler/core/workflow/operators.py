# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : didi & zhaoyang
# @Desc    : operator demo of aflow

import re
import random
import sys
import traceback
import multiprocessing
from collections import Counter
from typing import Dict, List, Tuple, Optional

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from workflow_compiler.core.llm.client import AsyncLLM
from workflow_compiler.core.logs import logger
from workflow_compiler.core.llm.formatter import BaseFormatter, FormatError, XmlFormatter, TextFormatter, CodeFormatter
from workflow_compiler.core.workflow.prompts import (
    ANSWER_GENERATION_PROMPT,
    FORMAT_PROMPT,
    MD_ENSEMBLE_PROMPT,
    PYTHON_CODE_VERIFIER_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT,
    REVIEW_PROMPT,
    REVISE_PROMPT,
)
from workflow_compiler.core.workflow.prompts import SC_ENSEMBLE_PROMPT2 as SC_ENSEMBLE_PROMPT
from workflow_compiler.core.utils.code import (
    extract_test_cases_from_jsonl,
    test_case_2_test_function,
)


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class FormatOp(BaseModel):
    solution: str = Field(default="", description="Your formatted answer for this problem")


class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class ReflectionTestOp(BaseModel):
    reflection_and_solution: str = Field(
        default="", description="Corrective solution for code execution errors or test case failures"
    )


class MdEnsembleOp(BaseModel):
    thought: str = Field(default="", description="Step-by-step analysis of the solutions to determine the best one.")
    solution_letter: str = Field(default="", description="The letter of the chosen best solution (only one letter).")


class ReviewOp(BaseModel):
    review_result: bool = Field(
        default=False,
        description="The Review Result (Bool). If you think this solution looks good for you, return 'true'; If not, return 'false'",
    )
    feedback: str = Field(
        default="",
        description="Your FeedBack for this problem based on the criteria. If the review result is true, you can put it 'nothing here'.",
    )


class ReviseOp(BaseModel):
    solution: str = Field(default="", description="Based on the feedback, revised solution for this problem")


class Operator:
    def __init__(self, llm: AsyncLLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, return_io_tokens=False, **extra_kwargs):
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        
        # Initialize token counters to track actual LLM usage
        input_tokens = 0
        output_tokens = 0
        
        try:
            # Use the formatter with AsyncLLM
            if formatter:
                # Prepare the prompt with formatting instructions
                formatted_prompt = formatter.prepare_prompt(prompt)
                
                # Call the LLM with the formatted prompt, requesting token info
                result = await self.llm(formatted_prompt, return_io_tokens=True)
                raw_response, input_tokens, output_tokens = result
                
                # Validate and parse the response
                is_valid, parsed_data = formatter.validate_response(raw_response)
                
                if not is_valid:
                    error_message = formatter.format_error_message()
                    raise FormatError(f"{error_message}.")
                
                # Add the raw response to the parsed data
                if isinstance(parsed_data, dict):
                    parsed_data["_raw_llm_output"] = raw_response
                    parsed_data["_raw_llm_prompt"] = formatted_prompt
                    parsed_data["_input_tokens"] = input_tokens
                    parsed_data["_output_tokens"] = output_tokens
                    return parsed_data if not return_io_tokens else (parsed_data, input_tokens, output_tokens)
                else:
                    result_dict = {"response": parsed_data, "_raw_llm_output": raw_response, "_raw_llm_prompt": formatted_prompt, "_input_tokens": input_tokens, "_output_tokens": output_tokens}
                    return result_dict if not return_io_tokens else (result_dict, input_tokens, output_tokens)
            else:
                # Fallback to direct call if no formatter is needed
                result = await self.llm(prompt, return_io_tokens=True)
                response, input_tokens, output_tokens = result
                
            # Convert to expected format based on the original implementation
            if isinstance(response, dict):
                response["_raw_llm_prompt"] = prompt
                response["_input_tokens"] = input_tokens
                response["_output_tokens"] = output_tokens
                return response if not return_io_tokens else (response, input_tokens, output_tokens)
            else:
                result_dict = {"response": response, "_raw_llm_prompt": prompt, "_input_tokens": input_tokens, "_output_tokens": output_tokens}
                return result_dict if not return_io_tokens else (result_dict, input_tokens, output_tokens)
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            error_dict = {"error": str(e)}
            # Return the actual token counts even when there's a format error
            return error_dict if not return_io_tokens else (error_dict, input_tokens, output_tokens)
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction, return_io_tokens=False):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill", return_io_tokens=return_io_tokens)
        
        if return_io_tokens:
            response_dict, input_tokens, output_tokens = response
            return response_dict, input_tokens, output_tokens
        else:
            return response


class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response


class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction, return_io_tokens=False):
        prompt = instruction.format(problem=problem, entry_point=entry_point)
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point, return_io_tokens=return_io_tokens)
        
        if return_io_tokens:
            response_dict, input_tokens, output_tokens = response
            return response_dict, input_tokens, output_tokens
        else:
            return response


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)
        self.prompt_type = "new" if "boxed" in SC_ENSEMBLE_PROMPT else "old"
    
    def _extract_boxed_answer(self, solution: str) -> str:
        """Extract answer from \boxed{...} in a solution."""
        match = re.search(r'\\boxed\{([^}]*)\}', solution)
        if match:
            return match.group(1).strip()
        return ""
    
    def _find_matching_solution(self, answer: str, solutions: List[str], answer_mapping: Dict[str, int]) -> int:
        """
        Find which solution matches the given answer.
        
        Args:
            answer: The answer to match (from sc_ensemble output)
            solutions: List of solution texts
            answer_mapping: Mapping from letters to solution indices
            
        Returns:
            Index of matching solution, or 0 (first solution) as fallback
        """
        # Extract boxed answers from each solution and compare
        for letter, idx in answer_mapping.items():
            solution_answer = self._extract_boxed_answer(solutions[idx])
            if solution_answer and solution_answer == answer:
                logger.warning(f"Matched answer '{answer}' with solution index {idx} (letter {letter})")
                return idx
        
        logger.warning(f"No matching solution found for answer: {answer}")
        # No match found, return first solution as fallback
        return 0

    async def __call__(self, solutions: List[str], problem: str, return_io_tokens=False):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text)
        if self.prompt_type == "new":
            response = await self._fill_node(ScEnsembleOp, prompt, mode="single_fill", return_io_tokens=return_io_tokens)
        else:
            response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill", return_io_tokens=return_io_tokens)
        
        if return_io_tokens:
            response_dict, input_tokens, output_tokens = response
            if self.prompt_type == "new":
                model_output = response_dict["response"]
                match = re.search(r'\\boxed\{([^}]*)\}', model_output)
                if match:
                    answer = match.group(1).strip()
                else:
                    answer = ""
            else:
                answer = response_dict.get("solution_letter", "")
                answer = answer.strip().upper()
            
            # Intelligent matching: if answer is not a valid option letter, try matching with solutions
            if answer in answer_mapping:
                selected_index = answer_mapping[answer]
            elif answer == "":
                # Empty answer, default to first solution
                logger.warning("Empty answer received from ScEnsemble, defaulting to first solution.")
                selected_index = 0
            else:
                # Answer is not a valid option letter, treat as potential answer
                # Try to match with solutions' boxed answers
                selected_index = self._find_matching_solution(answer, solutions, answer_mapping)
            
            selected_solution = solutions[selected_index]
            
            result = {
                "response": selected_solution,
                "_raw_llm_prompt": response_dict.get("_raw_llm_prompt", ""),
                "_raw_llm_output": response_dict.get("_raw_llm_output", ""),
                "_solution_letter": answer,
                "_selected_index": selected_index
            }
            return result, input_tokens, output_tokens
        else:
            raise NotImplementedError("Non-token return not implemented for ScEnsemble")


def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os",
            "sys",
            "subprocess",
            "multiprocessing",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "ggplot",
            "pylab",
            "tkinter",
            "PyQt5",
            "wx",
            "pyglet",
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info(f"Detected prohibited import: {lib}")
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if "solve" in global_namespace and callable(global_namespace["solve"]):
            result = global_namespace["solve"]()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"


def run_code_in_process(code, result_queue):
    """Module-level process target to keep multiprocessing picklable on spawn mode."""
    try:
        status, output = run_code(code)
        result_queue.put(("success", status, output))
    except Exception as e:
        result_queue.put(("error", str(e), ""))


class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer"):
        super().__init__(llm, name)

    async def exec_code(self, code, timeout=10):
        """
        Asynchronously execute code and return an error if timeout occurs.
        Uses multiprocessing for true isolation and timeout enforcement.
        """
        try:
            from multiprocessing import Process, Queue

            result_queue = Queue()
            process = Process(target=run_code_in_process, args=(code, result_queue))
            process.start()
            
            # Wait for process to complete with timeout
            process.join(timeout=timeout)
            
            if process.is_alive():
                # Process is still running after timeout - force kill it
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                return "Error", "Code execution timed out"
            
            # Get result from queue
            if not result_queue.empty():
                result_type, status, output = result_queue.get()
                if result_type == "error":
                    return "Error", f"Code execution error: {status}"
                return status, output
            else:
                return "Error", "Code execution failed: no result returned"
                
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem, analysis, feedback, mode, return_io_tokens=False):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(CodeGenerateOp, prompt, mode, return_io_tokens=return_io_tokens, function_name="solve")
        if return_io_tokens:
            response_dict, input_tokens, output_tokens = response
            response_dict["_feedback"] = feedback
            return response_dict, input_tokens, output_tokens
        else:
            response["_feedback"] = feedback
            return response

    async def __call__(self, problem: str, analysis: str = "None", return_io_tokens=False):
        """
        Call method, generate code and execute once without retry.
        """
        feedback = ""
        
        result = await self.code_generate(problem, analysis, feedback, mode="code_fill", return_io_tokens=True)
        code_response, input_tokens, output_tokens = result
        
        code = code_response["response"]
        if not code:
            result_dict = {"code": code, "output": "No code generated"}
            return (result_dict, input_tokens, output_tokens) if return_io_tokens else result_dict
        
        status, output = await self.exec_code(code)
        result_dict = {"code": code, "output": output}
        # Add raw prompt/output
        result_dict["_raw_llm_prompt"] = code_response.get("_raw_llm_prompt", "")
        result_dict["_raw_llm_output"] = code_response.get("_raw_llm_output", "")
        
        return (result_dict, input_tokens, output_tokens) if return_io_tokens else result_dict

class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)
        # Cache all test cases during initialization to avoid repeated file I/O
        self._test_cache = self._load_all_test_cases()
    
    def _load_all_test_cases(self):
        """Load all test cases from dataset files once during initialization"""
        from workflow_compiler.core.utils.code import CodeDataset
        import json
        import os
        
        cache = {}
        file_map = {
            CodeDataset.HUMAN_EVAL.value: "data/datasets/humaneval_public_test.jsonl",
            CodeDataset.MBPP.value: "data/datasets/mbpp_public_test.jsonl",
            CodeDataset.LIVE_CODE_BENCH.value: "data/ours/livecodebench_public_test.jsonl",
        }
        
        for dataset_name, file_path in file_map.items():
            cache[dataset_name] = {}
            if not os.path.exists(file_path):
                logger.warning(f"Test case file not found: {file_path}")
                continue
                
            try:
                key_field = "question_id" if dataset_name == CodeDataset.LIVE_CODE_BENCH.value else "entry_point"
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        key = data.get(key_field)
                        if key:
                            cache[dataset_name][key] = data.get("test")
            except Exception as e:
                logger.error(f"Error loading test cases from {file_path}: {e}")
        
        return cache
    
    def _get_test_cases(self, lookup_key, dataset):
        """Get test cases from cache with fallback to hardcoded cases"""
        from workflow_compiler.core.utils.code import CodeDataset
        
        # Normalize dataset to enum value
        dataset_value = dataset.value if isinstance(dataset, CodeDataset) else dataset
        
        # Hardcoded cases for specific entry points
        hardcoded_cases_map = {
            CodeDataset.HUMAN_EVAL.value: {
                "find_zero": "",
                "decode_cyclic": "",
                "decode_shift": "",
                "by_length": "",
                "add": "",
                "triangle_area": "",
                "correct_bracketing": "",
                "solve": "",
                "sum_squares": "",
                "starts_one_ends": "",
            },
            CodeDataset.MBPP.value: {
                "remove_odd": "",
                "replace_spaces": "",
                "snake_to_camel": "",
                "Split": "",
                "swap_List": "",
                "square_Sum": "",
                "sort_sublists": "",
                "unique_sublists": "",
            },
            CodeDataset.LIVE_CODE_BENCH.value: {},
        }
        
        # Check hardcoded cases first
        hardcoded_cases = hardcoded_cases_map.get(dataset_value, {})
        if lookup_key in hardcoded_cases:
            return hardcoded_cases[lookup_key]
        
        # Get from cache
        return self._test_cache.get(dataset_value, {}).get(lookup_key)

    def exec_code(self, solution, entry_point, dataset="MBPP", question_id=""):
        """Execute code against test cases. Supports MBPP, HumanEval, and LiveCodeBench."""
        from workflow_compiler.core.utils.code import CodeDataset
        
        # For LiveCodeBench, use question_id instead of entry_point to match test cases
        lookup_key = question_id if (dataset == CodeDataset.LIVE_CODE_BENCH.value or dataset == "LiveCodeBench") and question_id else entry_point
        
        test_cases = self._get_test_cases(lookup_key, dataset)
        
        if not test_cases:
            return {"exec_fail_case": f"No test cases found for entry_point: {entry_point}"}
        
        # For LiveCodeBench, test cases are in structured format {input, output}
        # We need to use a different execution method
        if dataset == CodeDataset.LIVE_CODE_BENCH.value or dataset == "LiveCodeBench":
            return self.exec_code_livecodebench(solution, test_cases, entry_point)
        
        # For MBPP/HumanEval, use the original assert-based testing
        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"
    
    def exec_code_livecodebench(self, solution, test_cases, entry_point):
        """Execute LiveCodeBench code with structured input/output test cases."""
        from workflow_compiler.core.utils.lcb_runner import run_test
        import json
        
        fail_cases = []
        timeout = 6  # Keep in sync with lcb_runner default
        
        # test_cases is now a list of dicts with 'input' and 'output' keys
        # Extract inputs and outputs directly
        inputs = []
        outputs = []
        
        for test_case in test_cases:
            if isinstance(test_case, dict) and 'input' in test_case and 'output' in test_case:
                inputs.append(test_case['input'])
                outputs.append(test_case['output'])
        
        if not inputs:
            return {"exec_fail_case": "No valid test cases for LiveCodeBench"}
        
        # Create a minimal sample for run_test
        # Only set fn_name for call-based problems (not stdio)
        # wrapped_function indicates stdio problem
        fn_name = None if entry_point == "wrapped_function" else entry_point
        
        sample = {
            "input_output": json.dumps({
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": fn_name
            })
        }
        
        def _run_lcb_test(sample_payload, solution_code, shared_result):
            # Run inside a short-lived process so reliability_guard mutations stay local
            try:
                res, meta = run_test(sample_payload, test=solution_code, debug=False, timeout=timeout)
                shared_result["result"] = res
                shared_result["metadata"] = meta
            except Exception as e:
                # Catch any unhandled exception and store error result
                shared_result["result"] = [-4]
                shared_result["metadata"] = {
                    "error_code": -4,
                    "error_message": f"Unexpected error in _run_lcb_test: {str(e)}"
                }

        try:
            with multiprocessing.Manager() as manager:
                shared = manager.dict()
                p = multiprocessing.Process(
                    target=_run_lcb_test,
                    args=(sample, solution, shared),
                )
                p.start()
                p.join(timeout=(timeout + 1) * len(inputs) + 5)

                if p.is_alive():
                    p.kill()
                    p.join()
                    return {"exec_fail_case": "LiveCodeBench execution timeout"}

                if "result" not in shared or "metadata" not in shared:
                    return {"exec_fail_case": "LiveCodeBench execution failed without result"}

                results = shared["result"]
                metadata = shared["metadata"]
            
            # Check if all tests passed
            if all(r == 1 for r in results):
                return "no error"

            # Collect failed test cases
            for i, result in enumerate(results):
                if result != 1:
                    error_info = {
                        "test_fail_case": {
                            "test_case": f"Input: {inputs[i]}, Expected: {outputs[i]}",
                            "error_type": "TestFailure",
                            "error_message": f"Test case {i} failed with result code: {result}",
                            "metadata": metadata,
                        }
                    }
                    fail_cases.append(error_info)
            return fail_cases
        except Exception as e:
            return {"exec_fail_case": f"LiveCodeBench execution error: {str(e)}"}
    
    async def __call__(self, problem, solution, entry_point, return_io_tokens=False, dataset="MBPP", question_id=""):
        """
        Test the solution with test cases once. Returns test result.
        
        Args:
            problem: The programming problem
            solution: The code solution to test
            entry_point: The function name/entry point
            return_io_tokens: Whether to return token counts (always 0 for testing)
            dataset: Dataset name (MBPP, HumanEval, LiveCodeBench)
            question_id: Question ID for LiveCodeBench
            
        Returns:
            If return_io_tokens=True: (result_dict, 0, 0)
            Otherwise: result_dict
            
            result_dict format:
            - If success: {"result": True, "solution": solution}
            - If failure: {"result": False, "solution": solution, "error": error_info}
        """
        result = self.exec_code(solution, entry_point, dataset=dataset, question_id=question_id)
        
        if result == "no error":
            result_dict = {"result": True, "solution": solution}
        elif isinstance(result, dict) and "exec_fail_case" in result:
            # Execution failure
            result_dict = {"result": False, "solution": solution, "error": result["exec_fail_case"], "error_type": "exec_failure"}
        else:
            # Test case failures
            result_dict = {"result": False, "solution": solution, "error": result, "error_type": "test_failure"}
        
        if return_io_tokens:
            return result_dict, 0, 0
        else:
            return result_dict


class ReflectionTest(Operator):
    """Reflect on test failures and generate improved code solution"""
    
    def __init__(self, llm: AsyncLLM, name: str = "ReflectionTest"):
        super().__init__(llm, name)
    
    async def __call__(self, problem, solution, error, error_type, entry_point, return_io_tokens=False):
        """
        Reflect on test/exec failures and generate an improved solution.
        
        Args:
            problem: The programming problem
            solution: The failed code solution
            error: Error information (string for exec failure, list of dicts for test failure)
            error_type: "exec_failure" or "test_failure"
            entry_point: The function name/entry point
            return_io_tokens: Whether to return token counts
            
        Returns:
            If return_io_tokens=True: (response_dict, input_tokens, output_tokens)
            Otherwise: response_dict
            
            response_dict contains:
            - response: The improved code solution
            - _raw_llm_prompt: The prompt sent to LLM
            - _raw_llm_output: The raw LLM response
            - _input_tokens: Number of input tokens
            - _output_tokens: Number of output tokens
        """
        # Format prompt based on error type
        if error_type == "exec_failure":
            prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                problem=problem,
                solution=solution,
                exec_pass=f"executed unsuccessfully, error: \n {error}",
                test_fail="executed unsucessfully",
            )
        else:  # test_failure
            prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                problem=problem,
                solution=solution,
                exec_pass="executed successfully",
                test_fail=error,
            )
        
        response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill", 
                                        return_io_tokens=True, function_name=entry_point)
        
        if return_io_tokens:
            response_dict, input_tokens, output_tokens = response
            return response_dict, input_tokens, output_tokens
        else:
            return response


class Format(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Format"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, mode: str = None):
        prompt = FORMAT_PROMPT.format(problem_description=problem, solution=solution)
        response = await self._fill_node(FormatOp, prompt, mode)
        return response


class Review(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Review"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, mode: str = None):
        prompt = REVIEW_PROMPT.format(problem=problem, solution=solution)
        response = await self._fill_node(ReviewOp, prompt, mode="xml_fill")
        return response


class Revise(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Revise"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, feedback, mode: str = None):
        prompt = REVISE_PROMPT.format(problem=problem, solution=solution, feedback=feedback)
        response = await self._fill_node(ReviseOp, prompt, mode="xml_fill")
        return response


class MdEnsemble(Operator):
    """
    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm: AsyncLLM, name: str = "MdEnsemble", vote_count: int = 5):
        super().__init__(llm, name)
        self.vote_count = vote_count

    @staticmethod
    def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, str]]:
        shuffled_solutions = solutions.copy()
        random.shuffle(shuffled_solutions)
        answer_mapping = {chr(65 + i): solutions.index(solution) for i, solution in enumerate(shuffled_solutions)}
        return shuffled_solutions, answer_mapping

    async def __call__(self, solutions: List[str], problem: str, mode: str = None):
        logger.info(f"solution count: {len(solutions)}")
        all_responses = []

        for _ in range(self.vote_count):
            shuffled_solutions, answer_mapping = self.shuffle_answers(solutions)

            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

            prompt = MD_ENSEMBLE_PROMPT.format(solutions=solution_text, question=problem)
            response = await self._fill_node(MdEnsembleOp, prompt, mode="xml_fill")

            answer = response.get("solution_letter", "A")
            answer = answer.strip().upper()

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        final_answer = solutions[most_frequent_index]
        return {"solution": final_answer}
