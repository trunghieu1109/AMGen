async def forward_93(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Extract the percentage of African American population from the passage and calculate the percentage of people who were not African American by subtracting from 100%."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting and calculating percentage not African American, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_reflect_instruction2 = "Subtask 2: Review and revise the initial calculation and reasoning to ensure accuracy and completeness of the percentage of people not African American."
    critic_instruction2 = "Please review the initial calculation and provide feedback on its accuracy and completeness."
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
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, reviewing and revising calculation, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results2['critic_agent'].id}, providing feedback, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining calculation, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    aggregate_instruction3 = "Subtask 3: Aggregate any variant answers from previous reasoning steps to select the most consistent and accurate percentage of people not African American."
    aggregate_desc3 = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=aggregate_desc3,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, aggregating variant answers, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    debate_instruction4 = "Subtask 4: Review the final aggregated answer for correctness and reliability before producing the final output."
    final_decision_instruction4 = "Subtask 4: Make final decision on the percentage of people not African American."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results3['thinking'], results3['answer']],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc4 = {
        'instruction': final_decision_instruction4,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        final_decision_desc=final_decision_desc4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, reviewing final aggregated answer, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, finalizing answer, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs