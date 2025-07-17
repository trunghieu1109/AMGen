async def forward_197(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and classify all given information including total cobalt concentration, "
        "thiocyanate concentration, and cumulative stability constants for all cobalt(II) thiocyanato complexes. "
        "Confirm the identity of the blue complex as the dithiocyanato species Co(SCN)2."
    )
    cot_agent_desc_stage0_sub1 = {
        'instruction': cot_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1
    )
    logs.append(log_stage0_sub1)

    debate_instruction_stage1_sub1 = (
        "Sub-task 1: Formulate the equilibrium expressions for the concentrations of all cobalt(II) species "
        "(free Co(II), Co(SCN)+, Co(SCN)2, Co(SCN)3-, Co(SCN)4 2-) using the cumulative stability constants and free SCN- concentration."
    )
    debate_desc_stage1_sub1 = {
        'instruction': debate_instruction_stage1_sub1,
        'context': ["user query", results_stage0_sub1.get('thinking', ''), results_stage0_sub1.get('answer', '')],
        'input': [taskInfo, results_stage0_sub1.get('thinking', ''), results_stage0_sub1.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage1_sub1, log_stage1_sub1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub1)

    debate_instruction_stage1_sub2 = (
        "Sub-task 2: Derive an expression for the free Co(II) concentration by applying the mass balance on total cobalt "
        "and the equilibrium expressions for all species."
    )
    debate_desc_stage1_sub2 = {
        'instruction': debate_instruction_stage1_sub2,
        'context': ["user query", results_stage0_sub1.get('thinking', ''), results_stage0_sub1.get('answer', ''),
                    results_stage1_sub1.get('thinking', ''), results_stage1_sub1.get('answer', '')],
        'input': [taskInfo, results_stage0_sub1.get('thinking', ''), results_stage0_sub1.get('answer', ''),
                  results_stage1_sub1.get('thinking', ''), results_stage1_sub1.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage1_sub2, log_stage1_sub2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    cot_sc_instruction_stage2_sub1 = (
        "Sub-task 1: Calculate the equilibrium concentrations of all cobalt species using the derived free Co(II) concentration "
        "and equilibrium expressions."
    )
    cot_sc_desc_stage2_sub1 = {
        'instruction': cot_sc_instruction_stage2_sub1,
        'input': [taskInfo, results_stage1_sub2.get('thinking', ''), results_stage1_sub2.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", results_stage1_sub2.get('thinking', ''), results_stage1_sub2.get('answer', '')]
    }
    results_stage2_sub1, log_stage2_sub1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc_stage2_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage2_sub1)

    debate_instruction_stage2_sub2 = (
        "Sub-task 2: Compute the percentage of the dithiocyanato cobalt(II) complex (Co(SCN)2) relative to the total cobalt concentration "
        "and compare with the given choices."
    )
    debate_desc_stage2_sub2 = {
        'instruction': debate_instruction_stage2_sub2,
        'context': ["user query", results_stage2_sub1.get('thinking', ''), results_stage2_sub1.get('answer', '')],
        'input': [taskInfo, results_stage2_sub1.get('thinking', ''), results_stage2_sub1.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2_sub2, log_stage2_sub2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_stage2_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub2)

    final_answer = await self.make_final_answer(results_stage2_sub2.get('thinking', ''), results_stage2_sub2.get('answer', ''))
    return final_answer, logs
