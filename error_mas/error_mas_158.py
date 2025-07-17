async def forward_158(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Analyze the observed spectral data to identify the rest-frame wavelength corresponding to the observed 790 nm peak and estimate the quasar's redshift, considering typical quasar emission lines and spectral features with context from the query."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_subtask_1, log1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Summarize and interpret the cosmological parameters (H_0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe) and their implications for the redshift-distance relation, with context from the query."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_subtask_2, log2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = "Sub-task 3: Compute the quasar's redshift from the observed wavelength (790 nm) and the assumed rest-frame wavelength derived in Stage 0 Subtask 1, with context from previous outputs and the query."
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results_subtask_1['thinking'], results_subtask_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_subtask_3, log3 = await self.sc_cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = "Sub-task 4: Calculate the comoving distance to the quasar using the Lambda-CDM cosmological model parameters (H_0=70, Omega_m=0.3, Omega_Lambda=0.7, flat universe) and the redshift computed in Subtask 3, with context from previous outputs and the query."
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results_subtask_2['thinking'], results_subtask_2['answer'], results_subtask_3['thinking'], results_subtask_3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results_subtask_4, log4 = await self.sc_cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_reflect_instruction5 = "Sub-task 5: Compare the calculated comoving distance from Subtask 4 with the given multiple-choice options (6,7,8,9 Gpc) and select the most appropriate value, providing reasoning."
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results_subtask_4['thinking'], results_subtask_4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"]
    }
    results_subtask_5, log5 = await self.reflexion(
        subtask_id="stage_2.subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results_subtask_5['thinking'], results_subtask_5['answer'])
    return final_answer, logs
