async def forward_180(self, taskInfo):
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1.1: Explicitly identify and verify the exact neutrino energy lines and continuous spectra from each proton-proton chain branch (pp-I, pp-II, pp-III), "
        "with particular emphasis on the precise placement of 7Be neutrino lines and 8B neutrino spectra within the 700-800 keV and 800-900 keV bands. "
        "Cross-check authoritative solar neutrino spectral data and confirm which branches contribute significantly or negligibly to each band, avoiding assumptions based on approximate energy ranges."
    )
    debate_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "final_decision_instruction": "Sub-task 1.1: Verify and finalize the spectral contributions of each branch to the specified energy bands.",
        "input": [taskInfo],
        "context_desc": ["user query"],
        "temperature": 0.5
    }
    results_1_1, log_1_1 = await self.debate(
        subtask_id="subtask_1_1",
        debate_desc=debate_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    cot_reflect_instruction_1_2 = (
        "Sub-task 1.2: Using the verified spectral data from subtask_1_1, quantitatively estimate the baseline neutrino flux contributions from each branch to the 700-800 keV and 800-900 keV bands before the hypothetical stoppage of the pp-III branch. "
        "Include integration or approximation of fluxes over the energy bands and explicitly confirm that pp-II neutrinos do not contribute to band 1, correcting previous flawed assumptions."
    )
    critic_instruction_1_2 = (
        "Please review and provide the limitations of the provided flux estimates, ensuring accuracy and consistency with spectral data from subtask_1_1."
    )
    reflexion_desc_1_2 = {
        "instruction": cot_reflect_instruction_1_2,
        "critic_instruction": critic_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of subtask 1.1", "answer of subtask 1.1"]
    }
    results_1_2, log_1_2 = await self.reflexion(
        subtask_id="subtask_1_2",
        reflect_desc=reflexion_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 2.1: Calculate the new neutrino fluxes in both energy bands after the sudden cessation of the pp-III branch, "
        "using the baseline fluxes from subtask_1_2. Explicitly consider that stopping pp-III removes neutrinos from both bands due to the continuous 8B spectrum tail. "
        "Debate and cross-validate the impact of pp-III stoppage on each band’s flux to avoid prior reasoning errors."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": "Sub-task 2.1: Finalize the recalculated neutrino fluxes in both bands after pp-III branch stoppage.",
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]],
        "context_desc": ["user query", "thinking of subtask 1.2", "answer of subtask 1.2"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="subtask_2_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    debate_instruction_2_2 = (
        "Sub-task 2.2: Compute the approximate ratio of fluxes Flux(band 1) / Flux(band 2) after the pp-III branch stops, "
        "based on the updated fluxes from subtask_2_1. Select the closest answer choice from the given options. "
        "Include a critical review of the ratio calculation and ensure the final conclusion is consistent with corrected spectral and flux assumptions."
    )
    debate_desc_2_2 = {
        "instruction": debate_instruction_2_2,
        "final_decision_instruction": "Sub-task 2.2: Finalize the flux ratio and select the closest answer choice.",
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "context_desc": ["user query", "thinking of subtask 2.1", "answer of subtask 2.1"],
        "temperature": 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="subtask_2_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2["thinking"], results_2_2["answer"])
    return final_answer, logs
