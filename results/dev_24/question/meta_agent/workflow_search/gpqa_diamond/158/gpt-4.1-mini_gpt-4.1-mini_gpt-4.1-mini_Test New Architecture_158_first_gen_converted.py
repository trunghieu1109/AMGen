async def forward_158(self, taskInfo):
    logs = []

    debate_instruction1 = (
        "Sub-task 1: Identify the rest-frame emission line corresponding to the observed spectral peak at 790 nm and calculate the quasar's redshift z. "
        "Consider common quasar emission lines and justify the redshift calculation based on the observed wavelength."
    )
    debate_desc1 = {
        'instruction': debate_instruction1,
        'final_decision_instruction': "Sub-task 1: Provide the most plausible rest-frame emission line and the calculated redshift z.",
        'input': [taskInfo],
        'context_desc': ["user query"],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Using the Lambda-CDM cosmological parameters (H0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe), "
        "compute the comoving distance to the quasar for the calculated redshift z from Sub-task 1. "
        "Consider multiple calculation approaches and ensure self-consistency in the results."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and accurate comoving distance value for the quasar given the redshift z and cosmological parameters."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
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
        "Sub-task 3: Assess the computed comoving distance from Sub-task 2 in the context of the given multiple-choice options (6, 7, 8, 9 Gpc). "
        "Debate the plausibility of each option and select the most reasonable comoving distance value."
    )
    final_decision_instruction3 = "Sub-task 3: Select the most plausible comoving distance value from the given options based on the computed result."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
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
