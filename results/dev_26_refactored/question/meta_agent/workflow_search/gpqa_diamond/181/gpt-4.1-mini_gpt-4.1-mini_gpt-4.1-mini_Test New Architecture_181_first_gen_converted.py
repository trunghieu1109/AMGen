async def forward_181(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and clarify the key parameters, variables, and physical context of the Mott-Gurney equation and the given statements to prepare for detailed analysis."
    )
    cot_agent_desc_stage0_sub1 = {
        "instruction": cot_instruction_stage0_sub1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.sc_cot(
        subtask_id="stage0_subtask1",
        cot_agent_desc=cot_agent_desc_stage0_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0_sub1)

    cot_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the physical assumptions and conditions underlying the Mott-Gurney equation, including device type, carrier traps, contact types, and dominant transport mechanisms."
    )
    cot_agent_desc_stage1_sub1 = {
        "instruction": cot_instruction_stage1_sub1,
        "input": [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"],
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent analysis of physical assumptions for the Mott-Gurney equation."
    }
    results_stage1_sub1, log_stage1_sub1 = await self.sc_cot(
        subtask_id="stage1_subtask1",
        cot_agent_desc=cot_agent_desc_stage1_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub1)

    cot_reflect_instruction_stage1_sub2 = (
        "Sub-task 2: Evaluate each of the four given statements against the established physical assumptions and conditions to determine their validity."
    )
    critic_instruction_stage1_sub2 = (
        "Please review and provide the limitations of provided evaluations of the four statements regarding the Mott-Gurney equation validity."
    )
    cot_reflect_desc_stage1_sub2 = {
        "instruction": cot_reflect_instruction_stage1_sub2,
        "critic_instruction": critic_instruction_stage1_sub2,
        "input": [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1", "thinking of stage1_subtask1", "answer of stage1_subtask1"]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id="stage1_subtask2",
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Select the statement that correctly describes the validity conditions of the Mott-Gurney equation based on the analysis from previous subtasks."
    )
    final_decision_instruction_stage2_sub1 = (
        "Sub-task 1: Provide the final answer indicating which statement is true about the validity of the Mott-Gurney equation."
    )
    debate_desc_stage2_sub1 = {
        "instruction": debate_instruction_stage2_sub1,
        "final_decision_instruction": final_decision_instruction_stage2_sub1,
        "input": [taskInfo, results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        "context": ["user query", "thinking of stage1_subtask2", "answer of stage1_subtask2"],
        "temperature": 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage2_subtask1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])

    return final_answer, logs
