async def forward_90(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    passage = taskInfo.content
    cot_instruction1 = "Subtask 1: Extract the death date of Cardinal Pierre des Près from the passage."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [passage],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting death date of Cardinal Pierre des Près, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_instruction2 = "Subtask 2: Extract the death date of Cardinal Pierre de Cros from the passage."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [passage],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting death date of Cardinal Pierre de Cros, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    cot_instruction3 = "Subtask 3: Compare the extracted death dates of Cardinal Pierre des Près and Cardinal Pierre de Cros to determine who died first."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [results1['answer'].content, results2['answer'].content],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 1", "answer of subtask 2"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, comparing death dates, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    final_answer_instruction = "Subtask 4: Generate the final answer stating who died first between Cardinal Pierre des Près and Cardinal Pierre de Cros, formatted as required."
    final_answer_desc = {
        'instruction': final_answer_instruction,
        'input': [results3['answer'].content],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 3"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=final_answer_desc
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, generating final answer, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs