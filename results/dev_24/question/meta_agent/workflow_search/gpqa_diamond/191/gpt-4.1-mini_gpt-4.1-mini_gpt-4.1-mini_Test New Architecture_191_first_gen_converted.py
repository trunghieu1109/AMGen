async def forward_191(self, taskInfo):
    logs = []

    debate_instruction_stage0 = "Sub-task 1: Analyze and classify the given physical elements and parameters: the spherical conductor, cavity, charge placement, and observation point, including vector definitions and geometric relationships."
    debate_desc_stage0 = {
        'instruction': debate_instruction_stage0,
        'final_decision_instruction': 'Sub-task 1: Provide a comprehensive classification and analysis of the physical setup and parameters.',
        'input': [taskInfo],
        'context_desc': ['user query'],
        'temperature': 0.5
    }
    results_stage0, log_stage0 = await self.debate(
        subtask_id='stage_0.subtask_1',
        debate_desc=debate_desc_stage0,
        n_repeat=self.max_round
    )
    logs.append(log_stage0)

    cot_sc_instruction_stage1 = (
        "Sub-task 1: Derive the expression for the electric field at point P outside the conductor by applying electrostatic principles, "
        "considering the conductor's influence, cavity displacement, and charge +q inside the cavity."
    )
    final_decision_instruction_stage1 = (
        "Sub-task 1: Synthesize and choose the most consistent and correct expression for the electric field at point P, "
        "based on the analysis from stage 0."
    )
    cot_sc_desc_stage1 = {
        'instruction': cot_sc_instruction_stage1,
        'final_decision_instruction': final_decision_instruction_stage1,
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_stage1, log_stage1 = await self.sc_cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_sc_desc_stage1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1)

    debate_instruction_stage2 = (
        "Sub-task 1: Extract and characterize the defining features of the derived electric field expression, "
        "focusing on the dependence on distances l, L, displacement s, and angle theta."
    )
    debate_desc_stage2 = {
        'instruction': debate_instruction_stage2,
        'final_decision_instruction': 'Sub-task 1: Provide a detailed characterization of the electric field expression features.',
        'input': [taskInfo, results_stage1['thinking'], results_stage1['answer'], results_stage0['thinking'], results_stage0['answer']],
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1'],
        'temperature': 0.5
    }
    results_stage2, log_stage2 = await self.debate(
        subtask_id='stage_2.subtask_1',
        debate_desc=debate_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2)

    cot_sc_instruction_stage3 = (
        "Sub-task 1: Select the correct formula for the magnitude of the electric field at point P from the given choices, "
        "based on the analysis and extracted features from previous stages."
    )
    final_decision_instruction_stage3 = (
        "Sub-task 1: Provide the final selection of the correct electric field formula from the given options."
    )
    cot_sc_desc_stage3 = {
        'instruction': cot_sc_instruction_stage3,
        'final_decision_instruction': final_decision_instruction_stage3,
        'input': [taskInfo, results_stage2['thinking'], results_stage2['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_stage3, log_stage3 = await self.sc_cot(
        subtask_id='stage_3.subtask_1',
        cot_agent_desc=cot_sc_desc_stage3,
        n_repeat=self.max_sc
    )
    logs.append(log_stage3)

    final_answer = await self.make_final_answer(results_stage3['thinking'], results_stage3['answer'])
    return final_answer, logs
