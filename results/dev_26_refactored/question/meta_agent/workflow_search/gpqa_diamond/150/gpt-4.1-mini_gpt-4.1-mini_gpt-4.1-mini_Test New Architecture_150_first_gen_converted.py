async def forward_150(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Normalize the given state vector and represent the observable operator P as a matrix in a suitable form for further processing. "
        "Use the provided state vector and matrix elements from the query."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent normalized state vector and matrix representation.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2_1 = (
        "Sub-task 2.1: Compute the eigenvalues and eigenvectors of the observable operator P to identify its spectral decomposition. "
        "Use the normalized matrix representation from Sub-task 1."
    )
    final_decision_instruction2_1 = "Sub-task 2.1: Decide the consistent eigenvalues and eigenvectors of P."
    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'final_decision_instruction': final_decision_instruction2_1,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        'temperature': 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    debate_instruction2_2 = (
        "Sub-task 2.2: Identify the eigenspace corresponding to the eigenvalue 0 and construct the projection operator onto this eigenspace. "
        "Use the eigenvalues and eigenvectors from Sub-task 2.1."
    )
    final_decision_instruction2_2 = "Sub-task 2.2: Decide the consistent projection operator onto the zero eigenvalue eigenspace."
    debate_desc2_2 = {
        'instruction': debate_instruction2_2,
        'final_decision_instruction': final_decision_instruction2_2,
        'input': [taskInfo, results2_1['thinking'], results2_1['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        'temperature': 0.5
    }
    results2_2, log2_2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc2_2,
        n_repeat=self.max_round
    )
    logs.append(log2_2)

    cot_sc_instruction2_3 = (
        "Sub-task 2.3: Project the normalized state vector onto the eigenspace of eigenvalue 0 using the projection operator. "
        "Use the normalized state vector from Sub-task 1 and the projection operator from Sub-task 2.2."
    )
    cot_sc_desc2_3 = {
        'instruction': cot_sc_instruction2_3,
        'final_decision_instruction': "Sub-task 2.3: Synthesize and choose the most consistent projected state vector.",
        'input': [taskInfo, results1['thinking'], results1['answer'], results2_2['thinking'], results2_2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results2_3, log2_3 = await self.sc_cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_sc_desc2_3,
        n_repeat=self.max_sc
    )
    logs.append(log2_3)

    debate_instruction3_1 = (
        "Sub-task 3.1: Calculate the probability of measuring the eigenvalue 0 by computing the squared magnitude of the projected state vector. "
        "Use the projected state vector from Sub-task 2.3."
    )
    final_decision_instruction3_1 = "Sub-task 3.1: Decide the consistent probability value for measuring eigenvalue 0."
    debate_desc3_1 = {
        'instruction': debate_instruction3_1,
        'final_decision_instruction': final_decision_instruction3_1,
        'input': [taskInfo, results2_3['thinking'], results2_3['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"],
        'temperature': 0.5
    }
    results3_1, log3_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc3_1,
        n_repeat=self.max_round
    )
    logs.append(log3_1)

    final_answer = await self.make_final_answer(results3_1['thinking'], results3_1['answer'])
    return final_answer, logs
