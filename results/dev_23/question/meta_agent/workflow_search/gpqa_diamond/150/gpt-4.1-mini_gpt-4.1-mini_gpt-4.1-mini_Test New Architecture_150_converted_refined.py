async def forward_150(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze and classify the given system state vector and the observable operator matrix P. Verify the Hermiticity (symmetry) of P and determine whether the state vector requires normalization before further calculations. This subtask must explicitly confirm these properties to avoid assumptions later."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = "Sub-task 2: Compute the characteristic polynomial of the operator matrix P and explicitly calculate all eigenvalues numerically. Verify whether 0 is an eigenvalue by solving the characteristic equation concretely, avoiding assumptions about eigenvalues."
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: For each eigenvalue found in Subtask 2, explicitly compute and output a normalized eigenvector. Provide the normalized eigenvector(s) corresponding to eigenvalue 0 if it exists. Produce concrete numeric eigenvectors to enable precise projection calculations."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = "Sub-task 4: Normalize the given system state vector explicitly, showing the normalization factor and the normalized vector components. This ensures correct probability calculation in subsequent steps and addresses the previous omission of normalization."
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Project the normalized system state vector (from Subtask 4) onto the eigenspace associated with eigenvalue 0 using the normalized eigenvector(s) from Subtask 3. Explicitly compute and output the projection vector components numerically, avoiding symbolic or abstract descriptions."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    debate_instruction6 = "Sub-task 6: Calculate the probability of measuring the eigenvalue 0 by computing the squared norm of the projection vector (from Subtask 5) divided by the squared norm of the original state vector (from Subtask 4). Include all numeric values and intermediate calculations to avoid speculative results."
    debate_desc6 = {
        'instruction': debate_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 4", "answer of subtask 4"]
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    cot_reflect_instruction7 = "Sub-task 7: Compare the calculated probability (from Subtask 6) with the given multiple-choice options and select the correct answer. Explicitly justify the choice based on numeric results rather than consensus or assumptions."
    critic_instruction7 = "Please review the probability calculation and choice selection, providing limitations if any."
    cot_reflect_desc7 = {
        'instruction': cot_reflect_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7, log7 = await self.reflexion(
        subtask_id="subtask_7",
        reflect_desc=cot_reflect_desc7,
        n_repeat=self.max_round
    )
    logs.append(log7)

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'])
    return final_answer, logs
