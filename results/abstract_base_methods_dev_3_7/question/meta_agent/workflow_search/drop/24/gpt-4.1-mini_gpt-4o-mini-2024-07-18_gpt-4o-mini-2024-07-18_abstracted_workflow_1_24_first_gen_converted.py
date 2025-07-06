async def forward_24(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Extract the number of Ethiopian troops mentioned in the passage relevant to the question."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting Ethiopian troops, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Extract the number of Cuban soldiers mentioned in the passage relevant to the question."
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
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting Cuban soldiers, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = "Subtask 3: Extract the number of Soviet advisors mentioned in the passage relevant to the question."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, extracting Soviet advisors, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_instruction4 = "Subtask 4: Compare the extracted numbers of Ethiopians, Cuban soldiers, and Soviet advisors to determine which group is the largest."
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, comparing group sizes, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction5 = "Subtask 5: Based on the comparison, produce the final answer indicating which group had the largest number, formatted as a single letter choice (A, B, or C)."
    final_decision_instruction5 = "Subtask 5: Make final decision on the largest group and format the answer as a single letter choice."
    debate_desc5 = {
        "instruction": debate_instruction5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "input": [taskInfo, results4['thinking'], results4['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc5 = {
        "instruction": final_decision_instruction5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        final_decision_desc=final_decision_desc5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, converting comparison to final choice, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
