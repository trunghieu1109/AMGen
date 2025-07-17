async def forward_180(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the neutrino energy spectra and flux contributions from each solar fusion branch, "
        "focusing on the pp-III branch and its characteristic neutrino energies relative to the 700-800 keV and 800-900 keV bands, "
        "with context from the user query."
    )
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

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, identify which neutrino energy bands (700-800 keV and 800-900 keV) "
        "receive significant contributions from the pp-III branch and which receive contributions from other branches, "
        "considering the user query and previous analysis."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, assess the impact of stopping the pp-III branch on the neutrino fluxes "
        "in the two energy bands, considering the 8.5-minute neutrino travel time and unchanged fluxes from other branches."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Derive the approximate flux values in each energy band after the pp-III branch stops, "
        "using the assessed impact and known relative flux contributions from previous subtasks."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Combine the derived flux values to calculate the approximate ratio Flux(band 1) / Flux(band 2) "
        "and select the closest answer choice from the provided options, based on outputs from Sub-task 2 and Sub-task 4."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results2['thinking'], results2['answer'], results4['thinking'], results4['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer'], results4['thinking'], results4['answer']],
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
