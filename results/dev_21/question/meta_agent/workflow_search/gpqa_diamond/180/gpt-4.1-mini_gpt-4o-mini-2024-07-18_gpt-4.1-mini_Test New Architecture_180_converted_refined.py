async def forward_180(self, taskInfo):
    logs = []

    cot_instruction_1_1 = (
        "Subtask 1_1: Retrieve and verify authoritative neutrino energy spectra and flux contributions "
        "for each solar fusion branch relevant to the problem (pp-I, pp-II, pp-III, Be-7, pep, B-8). "
        "Explicitly document known line energies, continuum ranges, and typical flux fractions within the 700-800 keV and 800-900 keV bands. "
        "This subtask addresses the critical failure of previous attempts that assumed spectral contributions without consulting actual data, "
        "by enforcing a fact-verification step to ground all subsequent reasoning in accurate physical information."
    )

    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }

    results_1_1, log_1_1 = await self.reflexion(
        subtask_id="subtask_1_1",
        reflect_desc=cot_agent_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    cot_sc_instruction_1_2 = (
        "Subtask 1_2: Using the verified spectral data from subtask_1_1, classify and clearly assign which solar fusion branches contribute significantly "
        "to each neutrino energy band (700-800 keV and 800-900 keV). Explicitly state the dominant contributors and quantify their relative flux contributions or fractions in each band. "
        "This step prevents the previous error of misclassifying branch contributions and ensures that the analysis is based on concrete spectral facts rather than assumptions."
    )

    cot_sc_desc_1_2 = {
        'instruction': cot_sc_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }

    results_1_2, log_1_2 = await self.sc_cot(
        subtask_id="subtask_1_2",
        cot_agent_desc=cot_sc_desc_1_2,
        n_repeat=self.max_sc
    )
    logs.append(log_1_2)

    cot_sc_instruction_2_1 = (
        "Subtask 2_1: Quantify the impact on neutrino fluxes in each energy band caused by the hypothetical stopping of the pp-III branch 8.5 minutes ago, "
        "using the classified spectral contributions from subtask_1_2. Calculate approximate numerical flux values or fractions before and after the pp-III branch stops, "
        "explicitly considering that other branches remain unchanged. This subtask addresses the previous failure to correctly estimate flux changes due to incorrect spectral assumptions."
    )

    cot_sc_desc_2_1 = {
        'instruction': cot_sc_instruction_2_1,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1_2", "answer of subtask 1_2"]
    }

    results_2_1, log_2_1 = await self.sc_cot(
        subtask_id="subtask_2_1",
        cot_agent_desc=cot_sc_desc_2_1,
        n_repeat=self.max_sc
    )
    logs.append(log_2_1)

    debate_instruction_3_1 = (
        "Subtask 3_1: Compute the approximate ratio of fluxes Flux(band 1) / Flux(band 2) after the pp-III branch stops, "
        "based on the flux values from subtask_2_1. Select the closest answer choice from the given options without biasing towards any specific answer. "
        "This subtask ensures that the final conclusion is directly supported by rigorously verified data and calculations, avoiding the previous error of premature or assumption-based conclusions."
    )

    debate_desc_3_1 = {
        'instruction': debate_instruction_3_1,
        'context': ["user query", "thinking of subtask 2_1", "answer of subtask 2_1"],
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }

    results_3_1, log_3_1 = await self.debate(
        subtask_id="subtask_3_1",
        debate_desc=debate_desc_3_1,
        n_repeat=self.max_round
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])

    return final_answer, logs
