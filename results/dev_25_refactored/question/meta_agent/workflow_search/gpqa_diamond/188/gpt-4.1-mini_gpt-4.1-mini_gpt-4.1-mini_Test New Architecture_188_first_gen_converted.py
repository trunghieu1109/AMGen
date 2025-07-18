async def forward_188(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize the physical context and definitions of each effective particle: Magnon, Skyrmion, Pion, and Phonon, "
        "with context from the given question and choices."
    )
    cot_agent_desc_stage0_sub1 = {
        "instruction": cot_instruction_stage0_sub1,
        "input": [taskInfo],
        "temperature": 0.5,
        "context": ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0_sub1)

    cot_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the type of symmetry (continuous, discrete, global, local) associated with each particle "
        "and determine if it arises from spontaneous symmetry breaking, based on the summary from stage_0.subtask_1."
    )
    final_decision_instruction_stage1_sub1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the symmetry breaking association of each particle."
    )
    cot_agent_desc_stage1_sub1 = {
        "instruction": cot_instruction_stage1_sub1,
        "final_decision_instruction": final_decision_instruction_stage1_sub1,
        "input": [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_stage1_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub1)

    cot_reflect_instruction_stage1_sub2 = (
        "Sub-task 2: Integrate the analysis of each particle’s symmetry breaking association to form a comparative understanding of their origins, "
        "based on outputs from stage_0.subtask_1 and stage_1.subtask_1."
    )
    critic_instruction_stage1_sub2 = (
        "Please review and provide the limitations of the provided solutions regarding the symmetry breaking associations of the particles."
    )
    cot_reflect_desc_stage1_sub2 = {
        "instruction": cot_reflect_instruction_stage1_sub2,
        "critic_instruction": critic_instruction_stage1_sub2,
        "input": [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Evaluate all particles against the criterion of spontaneous symmetry breaking and select the particle that is not associated with it, "
        "based on the integrated analysis from stage_1.subtask_2."
    )
    final_decision_instruction_stage2_sub1 = (
        "Sub-task 1: Select the particle not associated with spontaneous symmetry breaking."
    )
    debate_desc_stage2_sub1 = {
        "instruction": debate_instruction_stage2_sub1,
        "final_decision_instruction": final_decision_instruction_stage2_sub1,
        "input": [taskInfo, results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])

    return final_answer, logs
