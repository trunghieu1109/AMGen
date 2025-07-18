async def forward_150(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and summarize the given information about the system state and the observable matrix."
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

    cot_sc_instruction2 = "Sub-task 2: Determine the eigenvalues and eigenvectors of the observable matrix P based on the output from Sub-task 1."
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

    debate_instruction3 = "Sub-task 3: Project the system state onto the eigenvector corresponding to the eigenvalue 0 based on the output from Sub-task 2."
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

    cot_sc_instruction4 = "Sub-task 4: Calculate the probability of measuring the eigenvalue 0 using the projection from Sub-task 3."
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

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs