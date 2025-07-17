async def forward_196(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Extract and summarize the defining spectral features and structural information of compound X from the given IR and 1H NMR data."
    )
    cot_agent_desc_stage0 = {
        'instruction': cot_instruction_stage0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0, log_stage0 = await self.debate(
        subtask_id="stage0_subtask_1",
        debate_desc=cot_agent_desc_stage0,
        n_repeat=self.max_round
    )
    logs.append(log_stage0)

    cot_sc_instruction_stage1 = (
        "Sub-task 1: Assess the chemical transformation of compound X upon reaction with red phosphorus and HI, focusing on the expected functional group changes and their impact on the molecular structure."
    )
    cot_sc_desc_stage1 = {
        'instruction': cot_sc_instruction_stage1,
        'input': [taskInfo, results_stage0.get('thinking', ''), results_stage0.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage0_subtask_1", "answer of stage0_subtask_1"]
    }
    results_stage1, log_stage1 = await self.sc_cot(
        subtask_id="stage1_subtask_1",
        cot_agent_desc=cot_sc_desc_stage1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1)

    debate_instruction_stage2 = (
        "Sub-task 1: Evaluate and prioritize the given candidate final products by comparing their structures and substituents against the inferred product structure from the reaction and spectral data."
    )
    debate_desc_stage2 = {
        'instruction': debate_instruction_stage2,
        'context': ["user query", "thinking of stage0_subtask_1", "answer of stage0_subtask_1", "thinking of stage1_subtask_1", "answer of stage1_subtask_1"],
        'input': [taskInfo, results_stage0.get('thinking', ''), results_stage0.get('answer', ''), results_stage1.get('thinking', ''), results_stage1.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2, log_stage2 = await self.debate(
        subtask_id="stage2_subtask_1",
        debate_desc=debate_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2)

    final_answer = await self.make_final_answer(results_stage2.get('thinking', ''), results_stage2.get('answer', ''))
    return final_answer, logs
