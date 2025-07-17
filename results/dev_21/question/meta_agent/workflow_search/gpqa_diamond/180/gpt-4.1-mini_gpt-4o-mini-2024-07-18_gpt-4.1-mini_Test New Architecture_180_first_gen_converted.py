async def forward_180(self, taskInfo):
    logs = []

    debate_instruction_0_1 = "Subtask 0_1: Extract and summarize the defining features of the problem, including solar neutrino production branches, energy bands of interest, and the hypothetical scenario of stopping the pp-III branch 8.5 minutes ago."
    debate_desc_0_1 = {
        'instruction': debate_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_0_1, log_0_1 = await self.debate(
        subtask_id="subtask_0_1",
        debate_desc=debate_desc_0_1,
        n_repeat=self.max_round
    )
    logs.append(log_0_1)

    cot_sc_instruction_1_1 = "Subtask 1_1: Analyze and classify the neutrino energy spectra contributions from each solar fusion branch (pp-I, pp-II, pp-III) to the two energy bands (700-800 keV and 800-900 keV), based on the summary from Subtask 0_1."
    cot_sc_desc_1_1 = {
        'instruction': cot_sc_instruction_1_1,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 0_1", "answer of subtask 0_1"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="subtask_1_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    debate_instruction_1_2 = "Subtask 1_2: Determine the relative flux contributions of the pp-III branch neutrinos to each energy band and how their removal would affect the total flux in those bands, based on outputs from Subtask 0_1 and Subtask 1_1."
    debate_desc_1_2 = {
        'instruction': debate_instruction_1_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 0_1", "answer of subtask 0_1", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }
    results_1_2, log_1_2 = await self.debate(
        subtask_id="subtask_1_2",
        debate_desc=debate_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = "Subtask 2_1: Transform the classified spectral contributions into approximate numerical flux values or ratios for each band before and after stopping the pp-III branch, based on Subtask 1_1 and Subtask 1_2 outputs."
    debate_desc_2_1 = {
        'instruction': debate_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1", "thinking of subtask 1_2", "answer of subtask 1_2"]
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="subtask_2_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    debate_instruction_3_1 = "Subtask 3_1: Evaluate the flux ratio Flux(band 1) / Flux(band 2) after the pp-III branch stops and select the closest answer choice from the given options, based on Subtask 2_1 output."
    debate_desc_3_1 = {
        'instruction': debate_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2_1", "answer of subtask 2_1"]
    }
    results_3_1, log_3_1 = await self.debate(
        subtask_id="subtask_3_1",
        debate_desc=debate_desc_3_1,
        n_repeat=self.max_round
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
