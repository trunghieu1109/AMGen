async def forward_37(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Identify all parties Polin requested to join the Franco-Ottoman alliance from the passage provided in taskInfo."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying all parties Polin requested to join alliance, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Based on the parties identified in Subtask 1, determine which party or parties did not join the alliance despite Polin's request."
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
    agents.append(f"CoT agent {results2['cot_agent'].id}, determining party that did not join alliance, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    debate_instruction3 = "Subtask 3: Produce a concise and direct answer naming the party that did not join the alliance at Polin's request, based on Subtask 2's output."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc={
            'instruction': "Subtask 3: Make final decision on the party that did not join the alliance.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, producing concise answer, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, producing final concise answer, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs
