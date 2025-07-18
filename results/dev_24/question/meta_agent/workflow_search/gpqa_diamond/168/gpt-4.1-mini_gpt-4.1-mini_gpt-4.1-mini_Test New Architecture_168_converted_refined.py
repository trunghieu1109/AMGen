async def forward_168(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Extract and clearly summarize all given information about the decay processes, including particle types, masses (noting M is massless), number of emitted particles, and characteristics of the original energy spectrum of E particles. Clarify ambiguities such as whether the spectrum refers to total or individual E particle energies, and confirm the fixed total decay energy Q."
    final_decision_instruction1 = "Sub-task 1: Synthesize and choose the most consistent summary of the given decay information, ensuring clarity and completeness."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results_s1, log_s1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log_s1)

    cot_sc_instruction2 = "Sub-task 2: Formulate the conservation laws (energy and momentum) governing the original and variant decays, explicitly writing down the energy budget equations and constraints. Define variables for particle masses and energies, and establish the endpoint energy Q for the original decay quantitatively."
    final_decision_instruction2 = "Sub-task 2: Synthesize and choose the most consistent and correct formulation of conservation laws and endpoint definitions based on the summary from Sub-task 1."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results_s1['thinking'], results_s1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_s2, log_s2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log_s2)

    cot_reflect_instruction3 = "Sub-task 1: Perform a detailed quantitative kinematic analysis comparing the original decay (2A -> 2B + 2E + 2V) and the variant decay (2A -> 2B + 2E + M). Calculate or bound the maximum total energy available to the E particles in both cases, explicitly considering the massless nature of M and how it affects energy partitioning and phase space."
    critic_instruction3 = "Please review and provide the limitations of provided solutions for this quantitative kinematic analysis."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'critic_instruction': critic_instruction3,
        'input': [taskInfo, results_s1['thinking'], results_s1['answer'], results_s2['thinking'], results_s2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_s3, log_s3 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log_s3)

    cot_reflect_instruction4 = "Sub-task 2: Analyze how the replacement of two massive V particles by one massless M particle affects the continuity and shape of the total energy spectrum of the outgoing E particles. Confirm whether the spectrum remains continuous or becomes discrete, providing physical reasoning supported by the kinematic constraints derived earlier."
    critic_instruction4 = "Please review and provide the limitations of provided solutions for this spectrum continuity analysis."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results_s3['thinking'], results_s3['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_s4, log_s4 = await self.reflexion(
        subtask_id="stage_2.subtask_2",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log_s4)

    cot_sc_instruction5 = "Sub-task 1: Integrate the quantitative endpoint results and spectrum continuity conclusions to explicitly determine the direction and magnitude of the endpoint shift (Q' - Q). Derive a clear, quantitative formula or inequality showing whether the endpoint increases or decreases in the variant decay."
    final_decision_instruction5 = "Sub-task 1: Synthesize and choose the most consistent and correct conclusion about the endpoint shift based on previous analyses."
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results_s3['thinking'], results_s3['answer'], results_s4['thinking'], results_s4['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_s5, log_s5 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log_s5)

    cot_reflect_instruction6 = "Sub-task 2: Map the physics conclusions (spectrum continuity and endpoint shift) to the provided multiple-choice options. Explicitly identify which choice corresponds to the derived result and justify the selection with reference to the quantitative and qualitative findings."
    critic_instruction6 = "Please review and provide the limitations of the mapping and final answer selection."
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'critic_instruction': critic_instruction6,
        'input': [taskInfo, results_s5['thinking'], results_s5['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results_s6, log_s6 = await self.reflexion(
        subtask_id="stage_3.subtask_2",
        reflect_desc=cot_reflect_desc6,
        n_repeat=self.max_round
    )
    logs.append(log_s6)

    final_answer = await self.make_final_answer(results_s6['thinking'], results_s6['answer'])
    return final_answer, logs
