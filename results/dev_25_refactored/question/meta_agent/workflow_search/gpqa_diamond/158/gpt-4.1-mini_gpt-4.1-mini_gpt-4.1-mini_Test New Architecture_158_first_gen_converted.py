async def forward_158(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Estimate the redshift (z) of the quasar by relating the observed spectral peak wavelength (790 nm) "
        "to an assumed rest-frame wavelength, considering the flux drop at shorter wavelengths as an indicator of redshifted emission."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent redshift estimate.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Using the estimated redshift from Sub-task 1 and the given Lambda-CDM cosmological parameters "
        "(H_0=70 km/s/Mpc, Ω_m=0.3, Ω_Λ=0.7, flat universe), calculate the comoving distance to the quasar by integrating the inverse Hubble parameter over redshift."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent comoving distance calculation.",
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Compare the calculated comoving distance from Sub-task 2 with the provided multiple-choice options (6, 7, 8, 9 Gpc) "
        "and select the most reasonable value that matches the cosmological calculation and observational constraints."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': "Sub-task 3: Select the best matching comoving distance choice.",
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
