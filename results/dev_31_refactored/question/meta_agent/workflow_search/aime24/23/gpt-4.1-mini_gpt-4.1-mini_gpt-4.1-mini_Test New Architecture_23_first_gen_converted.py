async def forward_23(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Formulate variables representing each digit in the 2x3 grid and express the sum constraints for rows and columns as equations. "
        "Input: taskInfo containing the problem query and detailed analysis."
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

    cot_instruction_0_2 = (
        "Sub-task 2: Derive relationships and constraints linking digits across rows and columns based on place values and sum conditions. "
        "Input: taskInfo, thinking and answer from stage_0.subtask_1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Enumerate possible digit assignments for the grid cells that satisfy the row sum condition (sum of two 3-digit numbers equals 999). "
        "Input: taskInfo, thinking and answer from stage_0.subtask_2."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Filter enumerated assignments to those also satisfying the column sum condition (sum of three 2-digit numbers equals 99). "
        "Input: taskInfo, thinking and answer from stage_1.subtask_1."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 1: Verify each candidate digit assignment for consistency with all problem constraints and count valid solutions. "
        "Input: taskInfo, thinking and answer from stage_1.subtask_2."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Synthesize and select the valid digit assignments consistent with all constraints and count them."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=1
    )
    logs.append(log_2_1)

    cot_reflect_instruction_3_1 = (
        "Sub-task 1: Simplify and consolidate the count of valid digit assignments to produce the final answer. "
        "Input: taskInfo, thinking and answer from stage_2.subtask_1."
    )
    critic_instruction_3_1 = (
        "Please review and provide the limitations of provided solutions of valid digit assignments count consolidation."
    )
    cot_reflect_desc_3_1 = {
        "instruction": cot_reflect_instruction_3_1,
        "critic_instruction": critic_instruction_3_1,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc_3_1,
        n_repeat=2
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
