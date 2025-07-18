async def forward_193(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and summarize the problem setup: identify the spins, their possible values, the energy formula, "
        "and parameters (J, β). Clarify assumptions about spin interactions and configuration space. Ensure clear understanding "
        "of the problem context to avoid misinterpretation in later steps."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Enumerate all possible spin configurations (total 8) for the three spins and compute the corresponding "
        "energy values for each configuration using the given energy formula. Ensure accuracy in energy calculations to prevent propagation of errors."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask_1", "answer of subtask_1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Group the spin configurations by their distinct energy levels and determine the degeneracy (number of configurations) "
        "for each energy level. This grouping is critical for simplifying the partition function calculation and must be done carefully to avoid miscounting degeneracies."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': debate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask_2", "answer of subtask_2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Compute the partition function Z by summing over all configurations the Boltzmann factor e^{-βE}, "
        "using the grouped energies and their degeneracies to express Z compactly. Emphasize correctness in algebraic manipulation "
        "and expression simplification to ensure the derived formula is exact."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask_3", "answer of subtask_3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Perform a detailed, explicit term-by-term comparison between the derived partition function expression "
        "and each of the four given multiple-choice options, preserving the choice labels (A, B, C, D). Verify coefficients and exponents exactly "
        "to avoid misassignment of the correct choice label."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': debate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer'], results1['thinking']],
        'context_desc': ["user query", "thinking of subtask_4", "answer of subtask_4", "thinking of subtask_1"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_reflect_instruction6 = (
        "Sub-task 6: Conduct an independent verification loop (Reflexion) where multiple agents re-examine the matching process from subtask_5 "
        "to confirm the correct choice label. Reconcile any discrepancies and finalize the answer selection with high confidence."
    )
    critic_instruction6 = (
        "Please review and provide the limitations of provided solutions for this problem and confirm the final answer choice."
    )
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'critic_instruction': critic_instruction6,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask_1", "answer of subtask_1", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_3", "answer of subtask_3", "thinking of subtask_4", "answer of subtask_4", "thinking of subtask_5", "answer of subtask_5"]
    }
    results6, log6 = await self.reflexion(
        subtask_id="subtask_6",
        reflect_desc=cot_reflect_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
