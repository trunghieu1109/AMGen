async def forward_180(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Collect and compile authoritative solar neutrino spectral and flux data for all relevant branches "
        "of the solar fusion chain, including pp-I, pp-II, pp-III (8B), pep, and CNO cycle neutrinos (13N, 15O). "
        "Explicitly quantify the neutrino energy spectra and total fluxes at Earth for each branch, focusing on the energy ranges 700-800 keV and 800-900 keV. "
        "Provide a comprehensive, numerical basis for flux contributions in the specified bands."
    )
    debate_desc1 = {
        'instruction': cot_instruction1,
        'final_decision_instruction': 'Sub-task 1: Provide a detailed, authoritative compilation of solar neutrino spectral and flux data for all relevant branches, with numerical fluxes in the specified energy bands.',
        'input': [taskInfo],
        'context_desc': ['user query'],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id='subtask_1',
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Using the compiled spectral and flux data from Subtask 1, quantitatively map the neutrino flux contributions of each branch into the two energy bands (700-800 keV and 800-900 keV). "
        "Integrate the spectra over these bands to obtain numerical flux values per branch and identify dominant and subdominant contributors. "
        "Cross-validate spectral overlaps and provide precise numerical flux estimates."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and accurate numerical flux contributions per branch in each energy band, based on Subtask 1 data."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1'],
        'temperature': 0.5
    }
    results2, log2 = await self.sc_cot(
        subtask_id='subtask_2',
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Model the effect on neutrino fluxes in each energy band after the hypothetical stoppage of the pp-III branch, "
        "using the numerical flux contributions from Subtask 2. Subtract the pp-III flux component from each band while retaining all other branches unchanged, "
        "carefully accounting for residual fluxes from minor branches. Provide a self-consistent numerical recalculation of fluxes post-pp-III stoppage."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the recalculated neutrino fluxes in each energy band after pp-III branch stoppage, with numerical justification."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ['user query', 'thinking of subtask 2', 'answer of subtask 2'],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id='subtask_3',
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Compute the approximate ratio of fluxes Flux(band 1) / Flux(band 2) after the pp-III branch stops, "
        "using the recalculated fluxes from Subtask 3. Ensure the ratio reflects small but nonzero residual fluxes in band 1 due to minor branches, "
        "and provide a clear numerical justification of the final approximate ratio."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide the final approximate numerical ratio Flux(band 1) / Flux(band 2) after pp-III stoppage, with justification."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ['user query', 'thinking of subtask 3', 'answer of subtask 3'],
        'temperature': 0.5
    }
    results4, log4 = await self.sc_cot(
        subtask_id='subtask_4',
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
