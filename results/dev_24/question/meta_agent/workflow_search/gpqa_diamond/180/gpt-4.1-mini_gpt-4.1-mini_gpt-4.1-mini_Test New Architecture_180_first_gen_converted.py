async def forward_180(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the solar neutrino production branches, focusing on the pp-III branch and its characteristic neutrino energy spectrum, "
        "and identify which energy bands (700-800 keV and 800-900 keV) are predominantly influenced by which branches. "
        "Consider the proton-proton chain branches and their neutrino energy distributions in the context of the given query."
    )
    debate_desc1 = {
        'instruction': cot_instruction1,
        'final_decision_instruction': "Sub-task 1: Provide a classification and analysis of neutrino branches and their energy band influences.",
        'input': [taskInfo],
        'context_desc': ["user query"],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="stage0_subtask1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, summarize the neutrino flux contributions at Earth from each branch, "
        "emphasizing the relative flux magnitudes in the two specified energy bands (700-800 keV and 800-900 keV) and the effect of neutrino travel time (8.5 minutes delay). "
        "Use self-consistency to consider possible cases and uncertainties."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent summary of neutrino flux contributions in the two energy bands, "
        "given the analysis from Sub-task 1."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"],
        'temperature': 0.5
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage0_subtask2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Transform the classified neutrino flux data from Sub-tasks 1 and 2 into a comparative framework that models the flux in each energy band "
        "with and without the pp-III branch, quantifying the expected change in flux in each band due to the hypothetical stoppage. "
        "Consider that other branches remain unchanged."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a quantified model of flux changes in the two energy bands after stopping the pp-III branch."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1", "thinking of stage0_subtask2", "answer of stage0_subtask2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage1_subtask3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Compute the approximate ratio of fluxes Flux(band 1) / Flux(band 2) after the pp-III branch stops, "
        "using the transformed flux contributions from Sub-task 3 and considering that other branches remain unchanged. "
        "Use self-consistency to ensure robustness of the computed ratio."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide the final approximate ratio Flux(band 1) / Flux(band 2) after the pp-III branch stoppage."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of stage1_subtask3", "answer of stage1_subtask3"],
        'temperature': 0.5
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage2_subtask4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
