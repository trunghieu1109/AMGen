async def forward_188(self, taskInfo):
    logs = []

    cot_sc_instruction_stage0 = (
        "Sub-task 1: Extract and summarize the physical nature and origin of each effective particle (Magnon, Skyrmion, Pion, Phonon) "
        "with emphasis on their relation to spontaneous symmetry breaking, based on the given query."
    )
    cot_sc_desc_stage0 = {
        'instruction': cot_sc_instruction_stage0,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent summary for each particle's relation to spontaneous symmetry breaking.",
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

    cot_reflect_instruction_stage1 = (
        "Sub-task 1: Combine and integrate the summarized information from stage 0 to analyze the association of each particle "
        "with spontaneously-broken symmetries, highlighting which particles are Goldstone bosons or topological excitations arising from such breaking."
    )
    critic_instruction_stage1 = (
        "Please review and provide the limitations of the provided integrated analysis regarding the association of each particle "
        "with spontaneously-broken symmetries."
    )
    cot_reflect_desc_stage1 = {
        'instruction': cot_reflect_instruction_stage1,
        'critic_instruction': critic_instruction_stage1,
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1, log_stage1 = await self.reflexion(
        subtask_id="stage_1.subtask_1",
        reflect_desc=cot_reflect_desc_stage1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1)

    debate_instruction_stage2 = (
        "Sub-task 1: Based on the integrated analysis from stage 1, select the effective particle that is not associated with a spontaneously-broken symmetry. "
        "Provide reasoning supporting the choice."
    )
    final_decision_instruction_stage2 = (
        "Sub-task 1: Select the effective particle not associated with a spontaneously-broken symmetry based on the integrated analysis."
    )
    debate_desc_stage2 = {
        'instruction': debate_instruction_stage2,
        'final_decision_instruction': final_decision_instruction_stage2,
        'input': [taskInfo, results_stage1['thinking'], results_stage1['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        'temperature': 0.5
    }
    results_stage2, log_stage2 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2)

    final_answer = await self.make_final_answer(results_stage2['thinking'], results_stage2['answer'])

    return final_answer, logs
