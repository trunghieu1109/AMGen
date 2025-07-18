async def forward_150(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Normalize the given state vector to ensure it has unit length, preparing it for probability calculations. "
        "Use the state vector from the query and perform step-by-step reasoning."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Compute the eigenvalues and eigenvectors of the observable matrix P to identify its spectral decomposition. "
        "Use the matrix P from the query and perform step-by-step reasoning with self-consistency."
    )
    cot_agent_desc2 = {
        "instruction": cot_instruction2,
        "final_decision_instruction": "Sub-task 2: Synthesize and choose the most consistent eigenvalues and eigenvectors for the observable matrix P.",
        "input": [taskInfo],
        "temperature": 0.5,
        "context": ["user query"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Identify the eigenspace corresponding to the eigenvalue 0 by selecting the eigenvectors associated with eigenvalue 0. "
        "Use the eigenvalues and eigenvectors computed in Sub-task 2."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "final_decision_instruction": "Sub-task 3: Synthesize and select the eigenvectors corresponding to eigenvalue 0.",
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Project the normalized state vector onto the eigenspace of eigenvalue 0 to find the component of the state corresponding to that measurement outcome. "
        "Use the normalized state vector from Sub-task 1 and the eigenvectors from Sub-task 3."
    )
    cot_agent_desc4 = {
        "instruction": cot_instruction4,
        "final_decision_instruction": "Sub-task 4: Synthesize and compute the projection of the normalized state vector onto the eigenspace of eigenvalue 0.",
        "input": [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Calculate the probability of measuring the eigenvalue 0 by computing the squared magnitude of the projection obtained in Sub-task 4. "
        "Use the projection vector from Sub-task 4 and perform a debate to ensure correctness."
    )
    final_decision_instruction5 = "Sub-task 5: Provide the final probability of measuring eigenvalue 0 as a concise answer."
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4['thinking'], results4['answer']],
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
