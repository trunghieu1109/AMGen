async def forward_8(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Define the game states and characterize the winning and losing positions based on allowed moves (removing 1 or 4 tokens). "
        "Input content: problem description from taskInfo."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_1_1 = (
        "Sub-task 1: Evaluate the recursive structure of the game states to identify which positions are losing for the first player (Alice). "
        "Input content: results (thinking and answer) from stage_0.subtask_1 and problem description from taskInfo."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    review_instruction_2_1 = (
        "Sub-task 1: Enumerate all positive integers n ≤ 2024 and extract those for which the initial position is losing for Alice. "
        "Input content: results (thinking and answer) from stage_1.subtask_1 and problem description from taskInfo."
    )
    review_desc_2_1 = {
        "instruction": review_instruction_2_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id="stage_2.subtask_1",
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)

    sc_cot_instruction_3_1 = (
        "Sub-task 1: Analyze the extracted losing positions to count how many positive integers n ≤ 2024 allow Bob to guarantee a win. "
        "Input content: results (thinking and answer) from stage_1.subtask_1 and stage_2.subtask_1 and problem description from taskInfo."
    )
    final_decision_instruction_3_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for counting the number of positive integers n ≤ 2024 for which Bob has a guaranteed winning strategy."
    )
    sc_cot_desc_3_1 = {
        "instruction": sc_cot_instruction_3_1,
        "final_decision_instruction": final_decision_instruction_3_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=sc_cot_desc_3_1,
        n_repeat=self.max_sc
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
