async def forward_150(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Solve the characteristic polynomial det(P - lambda*I) = 0 for the observable matrix P "
        "to find all eigenvalues explicitly, including verifying the presence of eigenvalue 0. "
        "Provide numeric solution details to avoid abstract reasoning without concrete results."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Find and explicitly compute the eigenvectors corresponding to each eigenvalue found in subtask_1, "
        "especially the eigenvector(s) associated with eigenvalue 0. Normalize these eigenvectors numerically to ensure correct projection calculations later. "
        "Avoid skipping numeric eigenvector derivation and normalization."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_1", "answer of subtask_1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Normalize the given system state vector psi = (-1, 2, 1) to unit length, showing all numeric steps. "
        "This normalization is crucial for correct probability calculation and was previously overlooked or assumed."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Project the normalized state vector psi onto the normalized eigenvector(s) corresponding to eigenvalue 0, "
        "compute the inner product <v0|psi> explicitly with numeric values, and then calculate the squared magnitude of this projection. "
        "Emphasize that the probability is the squared magnitude (not the amplitude) to avoid conceptual errors."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_3", "answer of subtask_3"]
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=cot_agent_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_instruction5 = (
        "Sub-task 5: Verify that the computed probability from subtask_4 is a valid probability value between 0 and 1, "
        "and check that it matches one of the given answer choices. This verification step is added to catch conceptual or numeric inconsistencies before final answer selection."
    )
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask_4", "answer of subtask_4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_agent_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_instruction6 = (
        "Sub-task 6: Select the correct probability value from the given choices based strictly on the verified computed probability, "
        "ensuring no conceptual confusion between amplitude and probability. This final step must rely on the validated numeric result and not on assumptions or consensus without numeric backing."
    )
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_5", "answer of subtask_5"]
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=cot_agent_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
