async def forward_98(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Determine the original old age pension amount for a single person before the raise by subtracting the raise amount (300 francs) from the new pension amount (1,700 francs) using step-by-step reasoning."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, calculating original pension, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction2 = "Subtask 2: Review the calculated original pension amount for correctness and clarity."
    critic_instruction2 = "Please review the correctness and clarity of the original pension amount calculation."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, reviewing calculation, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback round {i}, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining review round {i}, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = "Subtask 3: Extract and format the final answer as the number of francs of the old age pension for a single person before the raise, without additional explanation."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, formatting final answer, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs