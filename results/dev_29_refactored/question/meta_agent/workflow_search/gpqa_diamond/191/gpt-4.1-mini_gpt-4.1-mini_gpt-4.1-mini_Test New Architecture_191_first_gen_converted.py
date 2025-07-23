async def forward_191(self, taskInfo):
    logs = []

    cot_instruction_0_0 = "Sub-task 0: Extract and summarize all given physical parameters and geometric relationships from the query."
    cot_agent_desc_0_0 = {
        'instruction': cot_instruction_0_0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_0_0, log_0_0 = await self.cot(
        subtask_id="stage_0.subtask_0",
        cot_agent_desc=cot_agent_desc_0_0
    )
    logs.append(log_0_0)

    cot_instruction_0_1 = "Sub-task 1: Analyze the physical setup to identify relevant electrostatic principles and relationships between variables."
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo, results_0_0['thinking'], results_0_0['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_sc_instruction_0_2 = "Sub-task 2: Document assumptions and clarify ambiguous aspects of the problem statement for further reasoning."
    final_decision_instruction_0_2 = "Sub-task 2: Synthesize and choose the most consistent assumptions and clarifications for the problem."
    cot_sc_desc_0_2 = {
        'instruction': cot_sc_instruction_0_2,
        'final_decision_instruction': final_decision_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)

    cot_instruction_1_0 = "Sub-task 0: Combine extracted information and physical analysis to form a consolidated understanding of the problem."
    cot_agent_desc_1_0 = {
        'instruction': cot_instruction_1_0,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_0, log_1_0 = await self.cot(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = "Sub-task 1: Apply electrostatic boundary conditions and conductor properties to deduce the effective charge distribution."
    cot_agent_desc_1_1 = {
        'instruction': cot_agent_instruction_1_1,
        'input': [taskInfo, results_1_0['thinking'], results_1_0['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_0", "answer of stage_1.subtask_0"]
    }
    results_1_1, log_1_1 = await self.answer_generate(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    aggregate_instruction_2_0 = "Sub-task 0: Evaluate each provided choice formula against the consolidated physical model for correctness."
    aggregate_desc_2_0 = {
        'instruction': aggregate_instruction_2_0,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_0, log_2_0 = await self.aggregate(
        subtask_id="stage_2.subtask_0",
        aggregate_desc=aggregate_desc_2_0
    )
    logs.append(log_2_0)

    cot_instruction_2_1 = "Sub-task 1: Select the formula that correctly represents the magnitude of the electric field at point P."
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo, results_2_0['thinking'], results_2_0['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_2.subtask_0", "answer of stage_2.subtask_0"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    debate_instruction_2_2 = "Sub-task 2: Assess the validity of the selected formula considering the geometry and electrostatic shielding effects."
    final_decision_instruction_2_2 = "Sub-task 2: Provide a final assessment and confirm the correctness of the selected formula."
    debate_desc_2_2 = {
        'instruction': debate_instruction_2_2,
        'final_decision_instruction': final_decision_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'context': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        'temperature': 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    formatter_instruction_3_0 = "Sub-task 0: Consolidate the selected formula and reasoning into a clear, formatted final answer."
    formatter_desc_3_0 = {
        'instruction': formatter_instruction_3_0,
        'input': [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"],
        'format': 'short and concise, without explaination'
    }
    results_3_0, log_3_0 = await self.specific_format(
        subtask_id="stage_3.subtask_0",
        formatter_desc=formatter_desc_3_0
    )
    logs.append(log_3_0)

    review_instruction_3_1 = "Sub-task 1: Review the final output for clarity, correctness, and completeness."
    review_desc_3_1 = {
        'instruction': review_instruction_3_1,
        'input': [taskInfo, results_3_0['thinking'], results_3_0['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_3.subtask_0", "answer of stage_3.subtask_0"]
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id="stage_3.subtask_1",
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])

    return final_answer, logs
