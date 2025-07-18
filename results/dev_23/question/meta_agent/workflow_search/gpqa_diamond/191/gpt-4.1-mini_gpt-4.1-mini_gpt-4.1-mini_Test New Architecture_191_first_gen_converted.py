async def forward_191(self, taskInfo):
    logs = []

    cot_instruction_stage0 = "Sub-task 1: Analyze and classify the given physical elements and parameters: the spherical conductor, cavity, charge placement, and vectors defining positions and angles, with context from taskInfo"
    cot_agent_desc_stage0 = {
        'instruction': cot_instruction_stage0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0, log_stage0 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0)

    debate_instruction_stage1 = "Sub-task 1: Assess the impact of placing a charge +q inside the off-center cavity on the conductor's charge distribution and resulting induced charges, based on output from stage_0.subtask_1"
    debate_desc_stage1 = {
        'instruction': debate_instruction_stage1,
        'context': ["user query", results_stage0['thinking'], results_stage0['answer']],
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage1, log_stage1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1)

    debate_instruction_stage2 = "Sub-task 1: Derive the expression for the electric field at point P outside the conductor considering the induced charges and the geometry of the cavity and charge placement, based on outputs from stage_0.subtask_1 and stage_1.subtask_1"
    debate_desc_stage2 = {
        'instruction': debate_instruction_stage2,
        'context': ["user query", results_stage0['thinking'], results_stage0['answer'], results_stage1['thinking'], results_stage1['answer']],
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer'], results_stage1['thinking'], results_stage1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2, log_stage2 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2)

    cot_sc_instruction_stage3 = "Sub-task 1: Combine and transform the quantitative inputs (distances L, l, s and angle theta) to simplify and identify the correct formula for the magnitude of the electric field at point P, based on output from stage_2.subtask_1"
    cot_sc_desc_stage3 = {
        'instruction': cot_sc_instruction_stage3,
        'input': [taskInfo, results_stage2['thinking'], results_stage2['answer']],
        'temperature': 0.5,
        'context': ["user query", results_stage2['thinking'], results_stage2['answer']]
    }
    results_stage3, log_stage3 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc_stage3,
        n_repeat=self.max_sc
    )
    logs.append(log_stage3)

    final_answer = await self.make_final_answer(results_stage3['thinking'], results_stage3['answer'])
    return final_answer, logs
