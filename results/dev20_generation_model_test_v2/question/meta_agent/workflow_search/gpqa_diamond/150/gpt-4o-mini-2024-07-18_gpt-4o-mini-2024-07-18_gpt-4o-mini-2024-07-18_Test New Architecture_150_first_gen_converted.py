async def forward_150(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []
    cot_instruction1 = "Sub-task 1: Extract the state vector and observable matrix from the given information."
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
    cot_sc_instruction2 = "Sub-task 2: Perform the matrix multiplication of the observable matrix with the state vector to find the resulting vector."
    N = self.max_sc
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
    debate_instruction_5 = "Sub-task 3: Calculate the probabilities of the measurement outcomes based on the resulting vector from the previous step."
    debate_desc5 = {
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log3)
    cot_reflect_instruction4 = "Sub-task 4: Identify the specific probability of measuring the observable value 0 and compare it with the provided choices."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs