CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Generate the complete function below:
"""

FIX_CODE_PROMPT = """
The provided solution failed to pass the tests. Please analyze the error and fix the code. Ensure the function name and signature remain unchanged. If necessary, add or modify imports, correct logical errors, and improve the implementation.

Problem:
{input}

Provide the corrected function below:
"""

GENERATE_TEST_CASES_PROMPT = """
Generate a set of diverse test cases for the given problem. Include edge cases, normal cases, and any special cases that might be relevant. Format the test cases as a Python list of tuples, where each tuple represents the input parameters for a single test case.

Problem:
{input}

Generate the test cases below:
"""