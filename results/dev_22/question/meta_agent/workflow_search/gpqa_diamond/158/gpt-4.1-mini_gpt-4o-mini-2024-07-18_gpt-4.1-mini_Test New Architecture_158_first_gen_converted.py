async def forward_158(self, taskInfo):
    logs = []

    cot_instruction0 = (
        "Sub-task 1: Analyze and classify the given observational data and cosmological parameters, "
        "including identifying the significance of the 790 nm spectral peak and the flux drop at shorter wavelengths, "
        "and summarizing the cosmological model assumptions."
    )
    debate_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query", "cosmological parameters", "spectral data"],
        'role': self.debate_role
    }
    results0, log0 = await self.debate(
        subtask_id="stage_0.subtask_1",
        debate_desc=debate_desc0,
        n_repeat=self.max_round
    )
    logs.append(log0)

    cot_sc_instruction1 = (
        "Sub-task 1: Derive the redshift of the quasar by associating the observed 790 nm peak "
        "with the known rest-frame Lyman-alpha emission line at 121.6 nm and calculating z = (lambda_observed / lambda_emitted) - 1."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 1: Compute the comoving distance to the quasar using the derived redshift "
        "and the given Lambda-CDM cosmological parameters (H_0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe) "
        "by integrating the inverse Hubble parameter over redshift."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 1: Select the comoving distance value from the provided options (6, 7, 8, 9 Gpc) "
        "that best matches the computed comoving distance, justifying the choice based on the calculation and cosmological context."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results2['thinking'], results2['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'role': self.debate_role
    }
    results3, log3 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
