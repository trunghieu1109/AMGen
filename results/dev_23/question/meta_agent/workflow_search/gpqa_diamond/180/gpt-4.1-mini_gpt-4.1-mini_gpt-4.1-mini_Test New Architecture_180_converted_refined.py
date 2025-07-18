async def forward_180(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Subtask 1: Explicitly enumerate and characterize the neutrino energy spectra produced by each solar fusion branch "
        "relevant to the problem (pp-I, pp-II, pp-III, 8B, 7Be). Include discrete neutrino lines and continuum spectra, "
        "with their standard-model flux values and spectral shapes. Avoid assuming a pp-II continuum in the 700-800 keV band. "
        "Provide a clear, quantitative spectral baseline for subsequent flux integration."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Subtask 2: Quantitatively integrate or estimate the neutrino flux contributions from each branch within the two specified energy bands (700-800 keV and 800-900 keV), "
        "using the spectral data enumerated in Subtask 1. Explicitly calculate or approximate the band-integrated fluxes, carefully accounting for discrete lines and continuum tails, "
        "especially the pp-III continuum tail overlapping the 700-800 keV band."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Subtask 3: Analyze the impact of hypothetically stopping the pp-III branch on the neutrino fluxes in the two energy bands, "
        "using the quantitative flux contributions from Subtask 2. Determine how removal of the pp-III continuum affects the 700-800 keV band flux and confirm that the 800-900 keV band flux remains largely unchanged due to dominance by the 7Be 0.862 MeV line."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results3, log3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Subtask 4: Derive the approximate ratio of neutrino fluxes Flux(700-800 keV) / Flux(800-900 keV) after the pp-III branch stops, "
        "based on the impact analysis in Subtask 3. Support the derivation with explicit numerical or order-of-magnitude estimates from previous subtasks, "
        "avoiding qualitative assumptions. Produce a robust, quantitatively justified ratio reflecting corrected spectral contributions and flux changes."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="stage_1.subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Stage 2 Subtask 1: Select the correct approximate ratio of fluxes from the given choices (0.1, 10, 1, 0.01) based on the derived ratio in stage_1.subtask_4. "
        "Justify the selection with clear reference to the quantitative spectral flux analysis and impact of stopping the pp-III branch. Ensure final choice is consistent with corrected reasoning and explicitly addresses previous errors."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
