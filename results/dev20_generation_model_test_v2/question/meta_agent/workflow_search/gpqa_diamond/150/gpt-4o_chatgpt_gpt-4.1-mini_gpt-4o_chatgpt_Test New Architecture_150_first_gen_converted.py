async def forward_150(self, taskInfo):
    logs = []

    # Stage 0: Extract and summarize given information
    cot_instruction1 = "Sub-task 1: Extract and summarize the given information about the system state and the observable matrix."
    cot_agent_desc1 = {
        'instruction': cot_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1", 
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    # Stage 0: Analyze the relationship between the system state and the observable
    cot_sc_instruction2 = "Sub-task 2: Analyze the relationship between the system state and the observable using matrix-vector multiplication."
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

    # Stage 1: Determine the eigenvalues and eigenvectors of the observable matrix P
    debate_instruction3 = "Sub-task 3: Determine the eigenvalues and eigenvectors of the observable matrix P."
    debate_desc3 = {
        "instruction": debate_instruction3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3", 
        debate_desc=debate_desc3, 
        n_repeat=self.max_round
    )
    logs.append(log3)

    # Stage 1: Project the system state vector onto the eigenvector corresponding to the eigenvalue of interest (0)
    debate_instruction4 = "Sub-task 4: Project the system state vector onto the eigenvector corresponding to the eigenvalue of interest (0) to prepare for probability calculation."
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

    # Stage 2: Compute the probability of obtaining the measurement outcome 0
    cot_sc_instruction5 = "Sub-task 5: Compute the probability of obtaining the measurement outcome 0 using the projection from the previous subtask."
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5, 
        'input': [taskInfo, results4['thinking'], results4['answer']], 
        'temperature': 0.5, 
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5", 
        cot_agent_desc=cot_sc_desc5, 
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs