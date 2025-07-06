async def forward_41(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Extract the percentage value of the Swedish ancestral group from the passage provided in taskInfo."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting Swedish percentage, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Extract the percentage value of the United States ancestral group from the passage, using the output from Subtask 1 as context."
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
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting United States percentage, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = "Subtask 3: Compare the extracted percentages of Swedish and United States ancestral groups to determine which is smaller, using outputs from Subtasks 1 and 2."
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
    agents.append(f"CoT agent {results3['cot_agent'].id}, comparing percentages, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction4 = "Subtask 4: Produce the final answer indicating which ancestral group is smaller, formatted as required by the query, using the comparison result from Subtask 3."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    final_results = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        final_decision_desc={
            'instruction': "Subtask 4: Make final decision on which ancestral group is smaller.",
            'output': ['thinking', 'answer'],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(final_results['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, producing final formatted answer, thinking: {final_results['list_thinking'][round][idx].content}; answer: {final_results['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, thinking: {final_results['thinking'].content}; answer: {final_results['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {final_results['thinking'].content}; answer - {final_results['answer'].content}")
    logs.append(final_results['subtask_desc'])

    final_answer = await self.make_final_answer(final_results['thinking'], final_results['answer'], sub_tasks, agents)
    return final_answer, logs
