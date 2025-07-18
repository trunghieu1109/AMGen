async def forward_158(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Identify the rest-frame spectral feature responsible for the observed flux drop at 790 nm, "
        "determine its rest-frame wavelength to enable redshift calculation, with context from the query."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent rest-frame spectral feature and wavelength.",
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
        "Sub-task 2: Based on the identified rest-frame wavelength from Sub-task 1, calculate the redshift (z) "
        "of the quasar using the observed wavelength 790 nm, with context from the query and Sub-task 1 output."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent redshift value.",
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

    cot_sc_instruction3 = (
        "Sub-task 3: Compute the comoving distance to the quasar at scale factor a=1 using the Lambda-CDM cosmological parameters "
        "(H_0=70 km/s/Mpc, Ω_m=0.3, Ω_Λ=0.7) and the redshift obtained in Sub-task 2, with context from the query and previous subtasks."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': "Sub-task 3: Synthesize and choose the most consistent comoving distance value.",
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the computed comoving distance from Sub-task 3 against the provided choices (6, 7, 8, 9 Gpc) "
        "and select the most plausible value based on the calculations and cosmological context."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': "Sub-task 4: Select the best matching comoving distance choice.",
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
