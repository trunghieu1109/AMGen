async def forward_160(self, taskInfo):
    logs = []

    cot_sc_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize all given physical parameters and conditions relevant to the mean free path determination, "
        "including vacuum pressure, temperature, electron beam presence, and initial mean free path λ1, based on the provided query."
    )
    cot_sc_desc_stage0_sub1 = {
        'instruction': cot_sc_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc_stage0_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0_sub1)

    cot_sc_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the classical mean free path λ1 based on gas kinetic theory and vacuum parameters, "
        "clarifying its dependence on pressure, temperature, and molecular properties, using outputs from stage_0.subtask_1."
    )
    cot_sc_desc_stage1_sub1 = {
        'instruction': cot_sc_instruction_stage1_sub1,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_stage1_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub1)

    cot_sc_instruction_stage1_sub2 = (
        "Sub-task 2: Analyze how the introduction of the electron beam modifies the effective mean free path to λ2 by considering electron-gas molecule scattering effects "
        "and their impact on transport properties, using outputs from stage_0.subtask_1 and stage_1.subtask_1."
    )
    cot_sc_desc_stage1_sub2 = {
        'instruction': cot_sc_instruction_stage1_sub2,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc_stage1_sub2,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Evaluate and compare λ2 relative to λ1 using the analyses from Stage 1, interpret the significance of the factor 1.22, "
        "and select the correct conclusion about the relationship between λ2 and λ1, based on outputs from stage_0.subtask_1, stage_1.subtask_1, and stage_1.subtask_2."
    )
    debate_desc_stage2_sub1 = {
        'instruction': debate_instruction_stage2_sub1,
        'context': [
            "user query",
            "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"
        ],
        'input': [
            taskInfo,
            results_stage0_sub1['thinking'], results_stage0_sub1['answer'],
            results_stage1_sub1['thinking'], results_stage1_sub1['answer'],
            results_stage1_sub2['thinking'], results_stage1_sub2['answer']
        ],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])
    return final_answer, logs
