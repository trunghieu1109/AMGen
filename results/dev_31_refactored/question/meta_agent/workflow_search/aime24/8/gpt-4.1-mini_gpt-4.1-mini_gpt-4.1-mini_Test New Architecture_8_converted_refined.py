async def forward_8(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Define the game states and characterize the winning and losing positions based on allowed moves (removing 1 or 4 tokens). "
        "Provide a clear explanation of the recursive structure of the game states and identify the pattern of losing positions modulo 5. "
        "Input content are the given query and detailed analysis from taskInfo."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Using the characterization from stage_0.subtask_1, rigorously prove that the losing positions for the first player (Alice) are exactly those where n congruent to 0 or 2 modulo 5. "
        "This subtask ensures correctness of the pattern before enumeration. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the proof of losing positions pattern."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_instruction_2_1 = (
        "Sub-task 1: Enumerate all positive integers n less than or equal to 2024 that satisfy the losing position condition n congruent to 0 or 2 modulo 5 based on the proven pattern. "
        "Explicitly use the correct counting formula: for residue r > 0, count = floor((N - r)/5) + 1; for r = 0, count = floor(N/5). "
        "This subtask addresses the previous off-by-one error by applying the correct enumeration method. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    review_instruction_2_2 = (
        "Sub-task 2: Verify the enumeration by cross-checking that the sum of counts over all five residue classes equals 2024, ensuring no counting errors. "
        "This verification step prevents off-by-one mistakes and confirms the correctness of the enumeration. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    review_desc_2_2 = {
        "instruction": review_instruction_2_2,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.review(
        subtask_id="stage_2.subtask_2",
        review_desc=review_desc_2_2
    )
    logs.append(log_2_2)

    cot_instruction_3_1 = (
        "Sub-task 1: Sum the counts of losing positions from stage_2.subtask_1 to find the total number of positive integers n less than or equal to 2024 for which Bob has a winning strategy. "
        "Incorporate the verification results from stage_2.subtask_2 to ensure the final count is accurate and free from previous enumeration errors. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1 & stage_2.subtask_2, respectively."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer'], results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_3_1, log_3_1 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
