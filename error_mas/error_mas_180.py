async def forward_180(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and summarize the defining features of the solar neutrino flux, the energy bands (700-800 keV and 800-900 keV), and the role of the pp-III branch in producing neutrinos within these bands, based on the given query."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = "Sub-task 2: Characterize the neutrino energy spectra from each proton-proton chain branch (pp-I, pp-II, pp-III), focusing on which branches contribute to the two energy bands (700-800 keV and 800-900 keV) and the relative flux magnitudes, using the output from Sub-task 1."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = "Sub-task 3: Analyze the impact of stopping the pp-III branch on the neutrino fluxes in the two energy bands (700-800 keV and 800-900 keV), considering the 8.5-minute neutrino travel time and the unchanged fluxes from other branches, based on outputs from Sub-tasks 1 and 2."
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = "Sub-task 4: Calculate or estimate the approximate ratio of fluxes Flux(band 1) / Flux(band 2) after the pp-III branch stops, based on the altered flux contributions and analysis from Sub-task 3."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Select the correct answer choice for the flux ratio Flux(band 1) / Flux(band 2) from the given options (0.1, 10, 1, 0.01) based on the calculated ratio from Sub-task 4 and justify the selection."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
