async def forward_165(self, taskInfo):
    logs = []

    cot_reflect_instruction0 = "Sub-task 1: Extract and classify all given quantitative and qualitative inputs from the Lagrangian, field content, vacuum expectation values (VEVs), and candidate mass formulas to form a coherent physical and mathematical framework. Explicitly identify all relevant fields, their quantum numbers, interactions, and the structure of the candidate formulas, ensuring clarity on notation and assumptions."
    cot_reflect_desc0 = {
        'instruction': cot_reflect_instruction0,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query']
    }
    results0, log0 = await self.reflexion(
        subtask_id='stage0_subtask1',
        reflect_desc=cot_reflect_desc0,
        n_repeat=self.max_round
    )
    logs.append(log0)

    cot_sc_instruction1 = "Sub-task 1: Derive the approximate mass formula for the pseudo-Goldstone boson H2 using effective potential and radiative correction methods. Explicitly track dependence on VEVs, loop contributions from all relevant particles including the top quark, and ensure mass dimension consistency at every step."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of stage0_subtask1', 'answer of stage0_subtask1']
    }
    results1, log1 = await self.sc_cot(
        subtask_id='stage1_subtask1',
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2_1 = "Sub-task 1: Perform detailed dimensional analysis of each candidate mass formula for H2. Compute mass dimension of each term, verify correct placement of VEV factors (x^2 + v^2) in numerator or denominator, and ensure overall formula dimension is mass squared (M^2). Immediately discard any candidate failing dimensional consistency."
    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'context': ['user query', 'thinking of stage0_subtask1', 'answer of stage0_subtask1', 'thinking of stage1_subtask1', 'answer of stage1_subtask1'],
        'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id='stage2_subtask1',
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    debate_instruction2_2 = "Sub-task 2: Evaluate physical consistency of each dimensionally consistent candidate formula by verifying presence and correct sign of all dominant loop contributions, especially top quark, gauge bosons, scalars, and singlet fermions. Cross-check coefficients' signs and inclusion of all relevant particles as per the Lagrangian and known radiative correction physics."
    debate_desc2_2 = {
        'instruction': debate_instruction2_2,
        'context': ['user query', 'thinking of stage0_subtask1', 'answer of stage0_subtask1', 'thinking of stage1_subtask1', 'answer of stage1_subtask1', 'thinking of stage2_subtask1', 'answer of stage2_subtask1'],
        'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer'], results2_1['thinking'], results2_1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results2_2, log2_2 = await self.debate(
        subtask_id='stage2_subtask2',
        debate_desc=debate_desc2_2,
        n_repeat=self.max_round
    )
    logs.append(log2_2)

    final_answer = await self.make_final_answer(results2_2['thinking'], results2_2['answer'])
    return final_answer, logs
