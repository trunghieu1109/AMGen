async def forward_160(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and summarize the given physical parameters and conditions, "
        "explicitly distinguishing between the mean free path of gas molecules (λ1) and the mean free path of electrons scattering off gas molecules (λ2). "
        "Embed feedback to avoid conflating these two distinct concepts, which was a key failure in previous reasoning."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_s1_st1, log_s1_st1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log_s1_st1)

    debate_instruction_s1_st2 = (
        "Sub-task 2: Derive the mean free path formula λ = 1/(nσ) for both gas molecules and electrons scattering off gas molecules. "
        "Calculate and interpret the ratio of molecular to electron scattering cross sections (σ_m/σ_e ≈ 1.22) to determine whether λ2 is greater or smaller than λ1. "
        "This subtask addresses the critical failure of misapplying the factor 1.22 and clarifies the physical basis of the factor as a multiplier, not a divisor."
    )
    final_decision_instruction_s1_st2 = (
        "Sub-task 2: Synthesize the debate and finalize the interpretation of the factor 1.22 and its effect on λ2 relative to λ1."
    )
    debate_desc_s1_st2 = {
        'instruction': debate_instruction_s1_st2,
        'final_decision_instruction': final_decision_instruction_s1_st2,
        'input': [taskInfo, results_s1_st1['thinking'], results_s1_st1['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        'temperature': 0.5
    }
    results_s1_st2, log_s1_st2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc_s1_st2,
        n_repeat=self.max_round
    )
    logs.append(log_s1_st2)

    debate_instruction_s2_st1 = (
        "Sub-task 1: Based on the derivation from stage_1.subtask_2, resolve the correct inequality or equality relationship between λ2 and λ1, "
        "explicitly debating and reflecting on the physical implications of electron scattering cross sections and mean free paths. "
        "Incorporate Reflexion to critically evaluate both sides of the argument and finalize the correct direction of the inequality, preventing groupthink."
    )
    final_decision_instruction_s2_st1 = (
        "Sub-task 1: Finalize the inequality relationship between λ2 and λ1 with critical reflection and debate synthesis."
    )
    debate_reflexion_desc_s2_st1 = {
        'instruction': debate_instruction_s2_st1,
        'final_decision_instruction': final_decision_instruction_s2_st1,
        'input': [taskInfo, results_s1_st2['thinking'], results_s1_st2['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        'temperature': 0.5
    }
    results_s2_st1, log_s2_st1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_reflexion_desc_s2_st1,
        n_repeat=self.max_round
    )
    logs.append(log_s2_st1)

    cot_reflect_instruction_s3_st1 = (
        "Sub-task 1: Map the derived quantitative relationship (e.g., λ2 = λ1 × 1.22) to the provided multiple-choice options. "
        "Explicitly verify consistency between the derived formula and the answer choices, preventing the previous failure of selecting an answer contradictory to the derivation. "
        "Embed feedback emphasizing the importance of final verification to avoid logical inconsistencies in answer selection."
    )
    critic_instruction_s3_st1 = (
        "Please review and provide the limitations of provided solutions for mapping the derived relationship to the multiple-choice answers."
    )
    cot_reflect_desc_s3_st1 = {
        'instruction': cot_reflect_instruction_s3_st1,
        'critic_instruction': critic_instruction_s3_st1,
        'input': [taskInfo, results_s1_st1['thinking'], results_s1_st1['answer'], results_s2_st1['thinking'], results_s2_st1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_s3_st1, log_s3_st1 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc_s3_st1,
        n_repeat=self.max_round
    )
    logs.append(log_s3_st1)

    final_answer = await self.make_final_answer(results_s3_st1['thinking'], results_s3_st1['answer'])
    return final_answer, logs
