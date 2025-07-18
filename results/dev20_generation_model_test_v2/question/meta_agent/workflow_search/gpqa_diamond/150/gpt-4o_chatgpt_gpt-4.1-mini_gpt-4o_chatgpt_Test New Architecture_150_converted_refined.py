async def forward_150(self, taskInfo):
    logs = []
    
    cot_instruction1 = "Sub-task 1: Extract and summarize the given information about the system state and the observable matrix, ensuring clarity on the matrix elements and the measurement outcome of interest."
    cot_agent_desc = {
        'instruction': cot_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1", 
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Normalize the system state vector and compute the eigenvalues and eigenvectors of the observable matrix P, focusing on the eigenvalue of interest (0). Ensure that the eigenvectors are normalized."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2, 
        'input': [taskInfo, results1['thinking'], results1['answer']], 
        'temperature': 0.5, 
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2", 
        cot_agent_desc=cot_sc_desc2, 
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = "Sub-task 3: Project the normalized system state vector onto the normalized eigenvector corresponding to the eigenvalue 0. Compute the inner product and prepare for probability calculation."
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3, 
        'input': [taskInfo, results2['thinking'], results2['answer']], 
        'temperature': 0.5, 
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3", 
        cot_agent_desc=cot_sc_desc3, 
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Calculate the probability of obtaining the measurement outcome 0 by evaluating the squared norm of the projection and dividing by the norm of the state vector. Verify the result against the provided choices."
    debate_desc4 = {
        "instruction": debate_instruction4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4", 
        debate_desc=debate_desc4, 
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs