async def forward_88(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Extract the percentage of free school meals for Southampton from the passage provided in taskInfo."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting Southampton free school meals percentage, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_instruction2 = "Subtask 2: Extract the percentage of free school meals for Wokingham from the passage provided in taskInfo."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting Wokingham free school meals percentage, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    diff_instruction = "Subtask 3: Calculate the difference in percentage points between Southampton and Wokingham free school meals percentages using the extracted values."
    diff_agent_desc = {
        'instruction': diff_instruction,
        'input': [results1['answer'].content, results2['answer'].content],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 1", "answer of subtask 2"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=diff_agent_desc
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, calculating difference, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs