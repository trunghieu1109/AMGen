async def forward_160(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Analyze and classify the given physical parameters and conditions: accelerating voltage, vacuum pressure, temperature, "
        "gas molecule presence, and initial mean free path λ1, with context from the provided query."
    )
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

    debate_instruction_stage1 = (
        "Sub-task 1: Assess the impact of initiating the electron beam on the scattering environment and how it modifies the effective mean free path from λ1 to λ2, "
        "considering electron-gas molecule interactions, based on the output from stage_0.subtask_1."
    )
    final_decision_instruction_stage1 = (
        "Sub-task 1: Debate and decide on the physical implications of the electron beam on the mean free path change from λ1 to λ2."
    )
    debate_desc_stage1 = {
        'instruction': debate_instruction_stage1,
        'final_decision_instruction': final_decision_instruction_stage1,
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        'temperature': 0.5
    }
    results_stage1, log_stage1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1)

    cot_instruction_stage2 = (
        "Sub-task 1: Derive the theoretical or empirical relationship between λ2 and λ1, incorporating the factor 1.22 and physical principles of electron scattering and gas kinetics, "
        "based on the output from stage_1.subtask_1."
    )
    final_decision_instruction_stage2 = (
        "Sub-task 1: Synthesize and choose the most consistent theoretical or empirical relationship between λ2 and λ1 involving the factor 1.22."
    )
    cot_sc_desc_stage2 = {
        'instruction': cot_instruction_stage2,
        'final_decision_instruction': final_decision_instruction_stage2,
        'input': [taskInfo, results_stage1['thinking'], results_stage1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_stage2, log_stage2 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc_stage2,
        n_repeat=self.max_sc
    )
    logs.append(log_stage2)

    cot_reflect_instruction_stage3 = (
        "Sub-task 1: Combine and transform the quantitative inputs (λ1, factor 1.22) and derived relationships to determine the correct inequality or equality describing λ2 relative to λ1, "
        "based on outputs from stage_0.subtask_1 and stage_2.subtask_1."
    )
    critic_instruction_stage3 = (
        "Please review and provide the limitations of provided solutions for determining the relationship between λ2 and λ1."
    )
    cot_reflect_desc_stage3 = {
        'instruction': cot_reflect_instruction_stage3,
        'critic_instruction': critic_instruction_stage3,
        'input': [
            taskInfo,
            results_stage0['thinking'], results_stage0['answer'],
            results_stage2['thinking'], results_stage2['answer']
        ],
        'temperature': 0.0,
        'context_desc': [
            "user query",
            "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1",
            "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"
        ]
    }
    results_stage3, log_stage3 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc_stage3,
        n_repeat=self.max_round
    )
    logs.append(log_stage3)

    final_answer = await self.make_final_answer(results_stage3['thinking'], results_stage3['answer'])
    return final_answer, logs
