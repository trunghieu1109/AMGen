async def forward_193(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Analyze and classify the given system elements: identify the spins, their possible values, "
        "the energy formula, and the parameters involved (J, β). Summarize the problem setup and clarify assumptions "
        "about spin interactions and configuration space, with context from the user query."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent and correct summary of the system elements and problem setup.",
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

    debate_instruction2 = (
        "Sub-task 2: Generate all possible spin configurations (total 8) for the three spins S1, S2, S3, "
        "each taking values +1 or -1, and compute the corresponding energy values for each configuration using the given energy formula E = -J(S1S2 + S1S3 + S2S3). "
        "Provide a detailed list of configurations and their energies."
    )
    final_decision_instruction2 = (
        "Sub-task 2: From the generated configurations and energies, finalize the complete list with correct energies for all 8 configurations."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Group the spin configurations generated in Sub-task 2 by their distinct energy levels, "
        "determine the degeneracy (number of configurations) for each energy value, and present the grouping clearly."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Finalize the grouping of configurations by energy and degeneracy, ensuring correctness and clarity."
    )
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

    cot_sc_instruction4 = (
        "Sub-task 4: Compute the partition function Z by summing over all configurations the Boltzmann factor e^{-βE}, "
        "using the grouped energies and their degeneracies from Sub-task 3 to simplify the sum. Provide the explicit expression for Z."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': "Sub-task 4: Synthesize and finalize the correct expression for the partition function Z.",
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Compare the computed partition function expression from Sub-task 4 with the given choices for Z, "
        "analyze the coefficients and exponents carefully, and identify the correct formula for Z among the provided options."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Finalize the correct choice for the partition function Z based on the comparison and analysis."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context_desc': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
