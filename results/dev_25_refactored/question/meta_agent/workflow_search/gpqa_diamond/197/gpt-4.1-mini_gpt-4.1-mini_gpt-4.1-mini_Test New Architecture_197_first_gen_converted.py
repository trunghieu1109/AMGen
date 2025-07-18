async def forward_197(self, taskInfo):
    logs = []

    cot_sc_instruction_stage0 = (
        "Sub-task 1: Extract and organize all given data: total cobalt concentration, SCN- concentration, "
        "and cumulative stability constants for all cobalt thiocyanato complexes. Confirm the identity of the blue dithiocyanato complex as Co(SCN)2."
    )
    cot_sc_desc_stage0 = {
        'instruction': cot_sc_instruction_stage0,
        'final_decision_instruction': "Sub-task 1: Synthesize and confirm extracted data and complex identity.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results_stage0, log_stage0 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc_stage0,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0)

    cot_debate_instruction_stage1_1 = (
        "Sub-task 1: Formulate the equilibrium expressions for each cobalt(II) thiocyanato complex "
        "using the cumulative stability constants and free ion concentrations. Express the concentration "
        "of each complex in terms of free Co(II) and SCN- concentrations."
    )
    cot_debate_desc_stage1_1 = {
        'instruction': cot_debate_instruction_stage1_1,
        'final_decision_instruction': "Sub-task 1: Provide the equilibrium expressions for all complexes.",
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        'temperature': 0.5
    }
    results_stage1_1, log_stage1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=cot_debate_desc_stage1_1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_1)

    cot_debate_instruction_stage1_2 = (
        "Sub-task 2: Set up and solve the mass balance equation for total cobalt concentration, "
        "summing free Co(II) and all complex species concentrations. Determine the free Co(II) concentration in solution."
    )
    cot_debate_desc_stage1_2 = {
        'instruction': cot_debate_instruction_stage1_2,
        'final_decision_instruction': "Sub-task 2: Provide the calculated free Co(II) concentration.",
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer'], results_stage1_1['thinking'], results_stage1_1['answer']],
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        'temperature': 0.5
    }
    results_stage1_2, log_stage1_2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=cot_debate_desc_stage1_2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_2)

    cot_debate_instruction_stage2_1 = (
        "Sub-task 1: Calculate the concentration of the dithiocyanato complex Co(SCN)2 using the free Co(II) concentration "
        "and SCN- concentration. Then compute its percentage relative to total cobalt concentration."
    )
    cot_debate_desc_stage2_1 = {
        'instruction': cot_debate_instruction_stage2_1,
        'final_decision_instruction': "Sub-task 1: Provide the percentage of the dithiocyanato complex.",
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer'], results_stage1_1['thinking'], results_stage1_1['answer'], results_stage1_2['thinking'], results_stage1_2['answer']],
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        'temperature': 0.5
    }
    results_stage2_1, log_stage2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=cot_debate_desc_stage2_1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_1)

    cot_debate_instruction_stage2_2 = (
        "Sub-task 2: Compare the calculated percentage of the dithiocyanato complex with the given answer choices and select the closest match."
    )
    cot_debate_desc_stage2_2 = {
        'instruction': cot_debate_instruction_stage2_2,
        'final_decision_instruction': "Sub-task 2: Select the closest matching answer choice.",
        'input': [taskInfo, results_stage2_1['thinking'], results_stage2_1['answer']],
        'context_desc': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        'temperature': 0.5
    }
    results_stage2_2, log_stage2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=cot_debate_desc_stage2_2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_2)

    final_answer = await self.make_final_answer(results_stage2_2['thinking'], results_stage2_2['answer'])
    return final_answer, logs
