async def forward_20(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Formulate the key equation (d1 + d0)^2 = d1 * b + d0 with digit constraints and interpret its implications for b-eautiful numbers. "
        "Input: taskInfo containing problem description and definitions."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Derive bounds and properties on digits d1, d0 and base b to limit the search space for solutions. "
        "Input: taskInfo, thinking and answer from stage_0.subtask_1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: For each base b starting from 2, enumerate all digit pairs (d1, d0) satisfying the equation and digit constraints, "
        "counting valid b-eautiful numbers. Input: taskInfo, thinking and answer from stage_0.subtask_2."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_instruction_2_1 = (
        "Sub-task 1: Analyze enumeration results to identify the smallest base b ≥ 2 for which the count of b-eautiful numbers exceeds ten. "
        "Input: taskInfo, thinking and answer from stage_0.subtask_2 and stage_1.subtask_1."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
