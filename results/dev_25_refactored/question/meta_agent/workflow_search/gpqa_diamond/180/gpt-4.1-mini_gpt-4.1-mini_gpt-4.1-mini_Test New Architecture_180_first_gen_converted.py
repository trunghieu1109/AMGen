async def forward_180(self, taskInfo):
    logs = []

    cot_sc_instruction_0_1 = (
        "Sub-task 0_1: Extract and clarify all given information and assumptions from the query, "
        "including the nature of the pp-III branch, neutrino energy bands, and the hypothetical scenario of stopping the pp-III branch 8.5 minutes ago."
    )
    final_decision_instruction_0_1 = (
        "Sub-task 0_1: Synthesize and choose the most consistent understanding of the problem context and assumptions."
    )
    cot_sc_desc_0_1 = {
        'instruction': cot_sc_instruction_0_1,
        'final_decision_instruction': final_decision_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results_0_1, log_0_1 = await self.sc_cot(
        subtask_id="subtask_0_1",
        cot_agent_desc=cot_sc_desc_0_1,
        n_repeat=self.max_sc
    )
    logs.append(log_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 0_2: Define and confirm the physical and astrophysical context, including neutrino production mechanisms in the Sun, "
        "neutrino travel time, and the meaning of flux in this problem, based on outputs from subtask 0_1."
    )
    final_decision_instruction_0_2 = (
        "Sub-task 0_2: Synthesize and choose the most consistent physical context and definitions for the problem."
    )
    cot_sc_desc_0_2 = {
        'instruction': cot_sc_instruction_0_2,
        'final_decision_instruction': final_decision_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 0_1", "answer of subtask 0_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="subtask_0_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)

    cot_sc_instruction_1_1 = (
        "Sub-task 1_1: Identify and characterize the neutrino energy spectra contributions from each proton-proton chain branch (pp-I, pp-II, pp-III) "
        "relevant to the 700-800 keV and 800-900 keV bands, based on outputs from subtask 0_2."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1_1: Synthesize and choose the most consistent characterization of neutrino spectra contributions in the specified energy bands."
    )
    cot_sc_desc_1_1 = {
        'instruction': cot_sc_instruction_1_1,
        'final_decision_instruction': final_decision_instruction_1_1,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 0_2", "answer of subtask 0_2"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="subtask_1_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_reflect_instruction_1_2 = (
        "Sub-task 1_2: Integrate the information on neutrino flux contributions from all branches to determine the baseline fluxes in band 1 and band 2 before the pp-III branch stops, "
        "based on outputs from subtask 1_1."
    )
    critic_instruction_1_2 = (
        "Please review and provide the limitations of provided solutions for integrating neutrino flux contributions and determining baseline fluxes."
    )
    cot_reflect_desc_1_2 = {
        'instruction': cot_reflect_instruction_1_2,
        'critic_instruction': critic_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }
    results_1_2, log_1_2 = await self.reflexion(
        subtask_id="subtask_1_2",
        reflect_desc=cot_reflect_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 2_1: Calculate the new fluxes in both energy bands after the pp-III branch stops, considering that only pp-III neutrinos vanish and others remain unchanged, "
        "based on outputs from subtask 1_2."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 2_1: Synthesize and choose the most consistent calculation of new fluxes after pp-III cessation."
    )
    debate_desc_2_1 = {
        'instruction': debate_instruction_2_1,
        'final_decision_instruction': final_decision_instruction_2_1,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'context_desc': ["user query", "thinking of subtask 1_2", "answer of subtask 1_2"],
        'temperature': 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="subtask_2_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    debate_instruction_2_2 = (
        "Sub-task 2_2: Compute the approximate ratio Flux(band 1) / Flux(band 2) after the pp-III branch cessation and select the closest answer choice from the given options, "
        "based on outputs from subtask 2_1."
    )
    final_decision_instruction_2_2 = (
        "Sub-task 2_2: Provide the final answer for the flux ratio and select the closest choice from the given options."
    )
    debate_desc_2_2 = {
        'instruction': debate_instruction_2_2,
        'final_decision_instruction': final_decision_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'context_desc': ["user query", "thinking of subtask 2_1", "answer of subtask 2_1"],
        'temperature': 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="subtask_2_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2['thinking'], results_2_2['answer'])

    return final_answer, logs
