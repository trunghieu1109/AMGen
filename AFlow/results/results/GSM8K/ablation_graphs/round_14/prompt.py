SOLVE_PROMPT = """
Solve the given math problem step by step. Show your work clearly and provide a final numerical answer.
"""

REVIEW_PROMPT = """
Review the initial solution to the math problem. Verify the calculations, ensure the reasoning is correct, and format the final answer as a single numerical value without any units or explanations. If there are any errors, correct them and provide the accurate result.
"""

SECOND_REVIEW_PROMPT = """
Perform a second review of the solution, focusing on common calculation errors and edge cases. Pay special attention to:
1. Unit conversions (e.g., minutes to hours, days to weeks)
2. Proper interpretation of the problem statement
3. Correct application of mathematical operations
4. Handling of fractions and decimals
5. Consistency in units throughout the solution

If you find any errors or inconsistencies, correct them and provide the accurate result. Present the final answer as a single numerical value without any units or explanations.
"""

ERROR_ANALYSIS_PROMPT = """
Analyze the current solution for potential errors, focusing on:
1. Unit conversion mistakes (e.g., forgetting to convert hours to minutes or days to weeks)
2. Misinterpretation of the problem statement (e.g., calculating for one month instead of two)
3. Calculation errors, especially with large numbers or multiple steps
4. Overlooking important details in the problem statement

If you identify any errors, provide a corrected solution with clear explanations. If no errors are found, confirm the current solution. In both cases, ensure the final answer is presented as a single numerical value without units or additional text.
"""

VERIFY_PROMPT = """
Extract the final numerical answer from the given solution. If the answer is already a single number, return it as is. If it's not, find the last numerical value in the text and return only that number without any additional text or formatting. Ensure the output is a single numerical value without any units, explanations, or additional characters.
"""