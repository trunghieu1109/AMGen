async def forward_158(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Systematically identify and interpret the spectral feature observed at 790 nm by explicitly distinguishing between emission lines "
        "(e.g., Lyman-alpha) and absorption breaks (e.g., Lyman limit or Gunn-Peterson trough). Enumerate possible rest-frame features, test each hypothesis against the observed spectral behavior "
        "(peak at 790 nm and flux drop at shorter wavelengths), and determine the most physically consistent redshift z."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent and correct redshift z based on the above analysis.",
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
        "Sub-task 2: Using the redshift z determined in Subtask 1 and the given Lambda-CDM cosmological parameters "
        "(H0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe), compute the comoving distance to the quasar at the present epoch (scale factor a=1). "
        "Ensure the calculation follows standard cosmological integral formulas and verify numerical accuracy."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent and correct comoving distance based on the above calculation.",
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

    cot_reflect_instruction3 = (
        "Sub-task 3: Compare the computed comoving distance from Subtask 2 against the provided multiple-choice options: "
        "choice1: 8 Gpc, choice2: 9 Gpc, choice3: 7 Gpc, choice4: 6 Gpc. Explicitly restate the options with their labels and numerical values to avoid ambiguity. "
        "Perform a reflexion step to cross-check and verify that the selected letter choice correctly corresponds to the numerical value closest to the computed distance. "
        "Ensure consistency and correctness in the final answer selection."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions for this problem, and confirm the final answer choice is consistent with the computed comoving distance."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'critic_instruction': critic_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
