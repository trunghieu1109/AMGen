async def forward_150(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Compute the eigenvalues and eigenvectors of the observable matrix P, and identify the eigenspace corresponding to eigenvalue 0 with context from taskInfo."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Normalize the given system state vector ψ to ensure proper probability calculation, based on taskInfo."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = "Sub-task 3: Project the normalized state vector ψ onto the eigenspace of eigenvalue 0 and calculate the squared magnitude of this projection, using outputs from subtask_1 and subtask_2."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=cot_agent_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = "Sub-task 4: Calculate the probability that the measurement of the observable P yields the eigenvalue 0 by normalizing the squared projection magnitude with respect to the norm of ψ, based on output from subtask_3."
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Select the correct probability value from the given choices based on the computed probability from subtask_4."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
