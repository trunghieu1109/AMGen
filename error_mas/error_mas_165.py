async def forward_165(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Extract and summarize the defining features of the given BSM Lagrangian, "
        "including field content, quantum numbers, vacuum expectation values, and the role of the pseudo-Goldstone boson H_2."
    )
    debate_desc_stage0 = {
        'instruction': cot_instruction_stage0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ['user query']
    }
    results_stage0, log_stage0 = await self.debate(
        subtask_id='stage0_subtask1',
        debate_desc=debate_desc_stage0,
        n_repeat=self.max_round
    )
    logs.append(log_stage0)

    cot_sc_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the impact of radiative corrections on the pseudo-Goldstone boson mass, "
        "focusing on the role of loop contributions from bosons and fermions, and the dependence on VEVs and particle masses."
    )
    cot_sc_desc_stage1_sub1 = {
        'instruction': cot_sc_instruction_stage1_sub1,
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of stage0_subtask1', 'answer of stage0_subtask1']
    }
    results_stage1_sub1, log_stage1_sub1 = await self.sc_cot(
        subtask_id='stage1_subtask1',
        cot_agent_desc=cot_sc_desc_stage1_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub1)

    cot_reflect_instruction_stage1_sub2 = (
        "Sub-task 2: Derive or recall the general form of the radiative mass correction formula for the pseudo-Goldstone boson "
        "in terms of the effective potential and identify the significance of each term and coefficient in the candidate formulas."
    )
    cot_reflect_desc_stage1_sub2 = {
        'instruction': cot_reflect_instruction_stage1_sub2,
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': [
            'user query',
            'thinking of stage0_subtask1', 'answer of stage0_subtask1',
            'thinking of stage1_subtask1', 'answer of stage1_subtask1'
        ]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id='stage1_subtask2',
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Evaluate and prioritize the given candidate formulas for the pseudo-Goldstone boson mass by comparing their structure, "
        "signs, included particle contributions, and normalization factors to identify the correct approximation."
    )
    debate_desc_stage2_sub1 = {
        'instruction': debate_instruction_stage2_sub1,
        'context': [
            'user query',
            results_stage1_sub1['thinking'], results_stage1_sub1['answer'],
            results_stage1_sub2['thinking'], results_stage1_sub2['answer']
        ],
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer'], results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id='stage2_subtask1',
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(
        results_stage2_sub1['thinking'],
        results_stage2_sub1['answer']
    )

    return final_answer, logs
