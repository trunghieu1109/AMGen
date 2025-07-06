async def forward_52(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Extract the number of scholars in the editorial board created in March 1773 from the passage provided in taskInfo."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting number of scholars, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Extract the number of scribes in the editorial board created in March 1773 from the passage, using the answer from Subtask 1 as context."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting number of scribes, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = "Subtask 3: Calculate how many more scribes than scholars were in the editorial board by subtracting the number of scholars from the number of scribes, using answers from Subtasks 1 and 2."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, calculating difference, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs