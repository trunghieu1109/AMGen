async def forward_150(self, taskInfo):
    logs = []

    cot_instruction0 = "Sub-task 1: Extract and summarize the defining features of the problem: the system's state vector and the observable operator matrix, including their dimensions and components."
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0, log0 = await self.debate(
        subtask_id="stage_0.subtask_1",
        debate_desc=cot_agent_desc0,
        n_repeat=self.max_round
    )
    logs.append(log0)

    cot_sc_instruction1_1 = "Sub-task 1: Normalize the given state vector to ensure it represents a valid quantum state."
    cot_sc_desc1_1 = {
        'instruction': cot_sc_instruction1_1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1_1,
        n_repeat=self.max_sc
    )
    logs.append(log1_1)

    cot_instruction1_2 = "Sub-task 2: Compute the eigenvalues and eigenvectors of the observable operator matrix P to identify the eigenspace corresponding to eigenvalue 0."
    cot_agent_desc1_2 = {
        'instruction': cot_instruction1_2,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_2, log1_2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=cot_agent_desc1_2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    cot_sc_instruction2_1 = "Sub-task 1: Select the eigenvector(s) associated with eigenvalue 0 and project the normalized state vector onto this eigenspace."
    cot_sc_desc2_1 = {
        'instruction': cot_sc_instruction2_1,
        'input': [taskInfo, results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results2_1, log2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2_1,
        n_repeat=self.max_sc
    )
    logs.append(log2_1)

    debate_instruction2_2 = "Sub-task 2: Calculate the probability of measuring the eigenvalue 0 by computing the squared norm of the projection obtained in the previous subtask."
    debate_desc2_2 = {
        'instruction': debate_instruction2_2,
        'input': [taskInfo, results2_1['thinking'], results2_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results2_2, log2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc2_2,
        n_repeat=self.max_round
    )
    logs.append(log2_2)

    final_answer = await self.make_final_answer(results2_2['thinking'], results2_2['answer'])
    return final_answer, logs
