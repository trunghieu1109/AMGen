async def forward_180(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize the defining features of solar neutrino production relevant to the problem, "
        "including energy ranges of neutrinos produced by each branch (pp-I, pp-II, pp-III), and timing of flux changes at Earth after stopping pp-III."
    )
    debate_desc_stage0_sub1 = {
        'instruction': cot_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.debate(
        subtask_id="stage0_subtask1",
        debate_desc=debate_desc_stage0_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage0_sub1)

    cot_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the impact of stopping the pp-III branch on the neutrino fluxes in the two specified energy bands (700-800 keV and 800-900 keV), "
        "considering which bands are dominated by pp-III neutrinos and which are not."
    )
    debate_desc_stage1_sub1 = {
        'instruction': cot_instruction_stage1_sub1,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.debate(
        subtask_id="stage1_subtask1",
        debate_desc=debate_desc_stage1_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub1)

    cot_sc_instruction_stage1_sub2 = (
        "Sub-task 2: Based on the output from Sub-task 1 and Stage 0, derive the approximate ratio of fluxes Flux(band 1) / Flux(band 2) after the pp-III branch stops, "
        "using analysis of neutrino energy spectra and flux contributions from each branch."
    )
    results_stage1_sub2, log_stage1_sub2 = await self.sc_cot(
        subtask_id="stage1_subtask2",
        cot_agent_desc={
            'instruction': cot_sc_instruction_stage1_sub2,
            'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
            'temperature': 0.5,
            'context': [
                "user query",
                "thinking of stage0_subtask1",
                "answer of stage0_subtask1",
                "thinking of stage1_subtask1",
                "answer of stage1_subtask1"
            ]
        },
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Select the correct approximate ratio of fluxes from the given choices (0.1, 10, 1, 0.01) based on the derived ratio and justify the selection."
    )
    debate_desc_stage2_sub1 = {
        'instruction': debate_instruction_stage2_sub1,
        'input': [taskInfo, results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage1_subtask2", "answer of stage1_subtask2"]
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage2_subtask1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])
    return final_answer, logs
