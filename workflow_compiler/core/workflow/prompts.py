# -*- coding: utf-8 -*-
# @Date    : 6/26/2024 17:07 PM
# @Author  : didi
# @Desc    : prompts of operators

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

FORMAT_PROMPT = """
For the question described as {problem_description},
please extract a short and concise answer contains only one word/few words from the following solution: {solution}.
Make sure there are no additional comments or explanations in your response.
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""



SC_ENSEMBLE_PROMPT2 = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

You are given several candidate solutions, each labeled with a letter (A, B, C, ...).

Your task is to:
1. Examine the *final answer choice* of each solution.
2. Identify which **option letter** (A, B, C, ...) appears most frequently.
3. Do NOT derive a new solution or restate the answer content itself.

Reasoning requirement:
- Provide a **very brief reason (1–2 sentences)** explaining *which option appears most often*.
- The reason should ONLY discuss frequency or consistency across options.
- Do NOT explain the problem, logic, or correctness of the answer.

Important constraints:
- You must select **one of the existing option letters only**.
- Do NOT output the answer text itself.
- If there is a tie, choose the option that appears first alphabetically.

Output format:
- First, output the brief reason as plain text.
- Then, on a new line, output the final answer strictly in the format:

\\boxed{{<OPTION_LETTER>}}

where <OPTION_LETTER> is a single capital letter (e.g., A, B, C).
"""



PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""


REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.
"""

MD_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the solution that is more capable of solving the problem compared to other solutions, as this is crucial for problem-solving.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

REVIEW_PROMPT = """
Given a problem and a thoughtful solution, your task is to using critical thinking (questioning) to review the solution's correctness and provide a review result in boolean format.

problem: {problem}
solution: {solution}

If you are more than 95 percent confident that the final answer is incorrect, please return False and give a feedback for the error. Otherwise, please return True and give a explanation for the correctness.
"""

REVISE_PROMPT = """
Given a problem and a thoughtful solution which is just reviewed as incorrect, your task is to revise the solution to solve the question and ensure the final code solution is wrapped with ```python```.

problem: {problem}
solution: {solution}
feedback: {feedback}

Ensure the output code is self-contained, and without any additional text or test cases.
"""

# ============================================================================
# Simple Math Solver Prompt (for GSM8K and similar grade-school math problems)
# ============================================================================
# This prompt is designed to be easily customizable. Modify it to change how
# the simple one-agent workflow solves math problems.

SIMPLE_MATH_SOLVE_PROMPT = """
Solve the following math problem step by step.

Problem: {problem}

Instructions:
1. Read the problem carefully and identify what is being asked.
2. Break down the problem into clear, logical steps.
3. Show your calculations for each step.
4. Provide the final numerical answer.

Format your response as follows:
- First, show your step-by-step reasoning and calculations.
- Then, provide your final answer enclosed in \\boxed{{}}.

For example, if the answer is 42, write: \\boxed{{42}}
"""

# ============================================================================
# Default Workflow Prompts (Fallbacks for missing external prompt packages)
# ============================================================================

# These are lightweight defaults used when external prompt files are unavailable.
# They are designed to keep the workflows runnable in open-source settings.

GENERATE_SOLUTION_PROMPT = (
    "Solve the following problem step by step and provide the final answer.\\n\\n"
    "Problem:\\n"
)

DETAILED_SOLUTION_PROMPT = (
    "Provide a detailed, step-by-step solution with clear reasoning.\\n\\n"
    "Problem:\\n"
)

REFINE_ANSWER_PROMPT = (
    "You are given a draft solution. Refine it for correctness and clarity.\\n\\n"
    "Input:\\n"
)

FORMAT_ANSWER_PROMPT = (
    "Format the final answer concisely without extra explanation.\\n\\n"
    "Input:\\n"
)

CODE_GENERATE_PROMPT = """
Write a correct Python solution for the following coding problem.

Problem:
{problem}

If an entry point is specified, implement the function with that name.
Return only the code.
"""
