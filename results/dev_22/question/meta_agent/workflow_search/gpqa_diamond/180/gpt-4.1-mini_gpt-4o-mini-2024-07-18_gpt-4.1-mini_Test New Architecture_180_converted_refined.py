async def forward_180(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Subtask 1: Extract and tabulate the standard solar model neutrino energy spectra and relative flux contributions "
        "from each solar fusion branch (pp-I, pp-II, pp-III, pep, 7Be, 8B). Provide quantitative or semi-quantitative data on neutrino fluxes "
        "and their energy distributions, especially focusing on the 700-800 keV and 800-900 keV energy bands. "
        "Use debate agent collaboration to ensure accuracy and coverage."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Subtask 2: Using the extracted data from Subtask 1, accurately classify which solar fusion branches contribute significantly "
        "to the 700-800 keV and 800-900 keV neutrino energy bands. Correct any misattributions, especially clarifying the pp-III branch's minor contribution. "
        "Apply self-consistency chain-of-thought agent collaboration to ensure robust classification."
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

    cot_reflect_instruction3 = (
        "Subtask 3: Assess the impact of hypothetically stopping the pp-III branch on the neutrino fluxes in the 700-800 keV and 800-900 keV bands, "
        "using the verified spectral and flux data from Subtasks 1 and 2. Incorporate the 8.5-minute neutrino travel time delay and that all other branches continue unchanged. "
        "Use reflexion agent collaboration to filter and validate the impact scenarios."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="stage_2.subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Subtask 4: Derive approximate numerical flux values in each energy band after the pp-III branch stops, "
        "based on the impact assessment from Subtask 3 and the relative flux contributions from Subtask 2. "
        "Use self-consistency chain-of-thought agent collaboration to ensure quantitative accuracy."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_2.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Subtask 5: Calculate the approximate ratio Flux(band 1) / Flux(band 2) after stopping the pp-III branch by combining the derived flux values from Subtask 4. "
        "Select the closest answer choice from the given options. Verify that the ratio reflects the physical expectation that band 2 flux drops significantly while band 1 flux remains largely unchanged. "
        "Use debate agent collaboration to finalize and validate the answer."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of stage_2.subtask_4", "answer of stage_2.subtask_4", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        'input': [taskInfo, results4['thinking'], results4['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage_3.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
