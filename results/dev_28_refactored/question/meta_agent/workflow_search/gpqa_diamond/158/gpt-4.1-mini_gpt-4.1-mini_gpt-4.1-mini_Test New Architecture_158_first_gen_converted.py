async def forward_158(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Estimate the redshift (z) of the quasar by relating the observed wavelength peak (790 nm) "
        "to a plausible rest-frame wavelength of a known quasar emission feature, considering common emission lines and their typical rest wavelengths."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent redshift estimate.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Construct the cosmological model framework using the given Lambda-CDM parameters "
        "(H_0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe) to express the comoving distance as an integral function of redshift z. "
        "Use the redshift estimate from Sub-task 1 as input."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent cosmological integral framework.",
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Apply the cosmological integral formula constructed in Sub-task 2 to compute the comoving distance "
        "for the estimated redshift from Sub-task 1. Perform numerical integration as needed."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': "Sub-task 3: Synthesize and choose the most consistent computed comoving distance.",
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Evaluate the computed comoving distance from Sub-task 3 against the provided multiple-choice options (6, 7, 8, 9 Gpc). "
        "Critically assess the consistency and select the best matching candidate."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of the computed comoving distance and the reasoning for selecting the best matching option."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="stage_4.subtask_1",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
