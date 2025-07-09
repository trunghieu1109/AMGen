async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = "Subtask 1: Formulate the system of equations for Aya's walking and coffee-shop problem, clearly distinguishing walking time and coffee shop time. Validate the correctness of the equations against the problem statement."
    critic_instruction1 = "Please review the formulated system of equations and provide feedback on any errors or ambiguities."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, formulating and validating system of equations, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correctness: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining equations, thinking: {results1['list_thinking'][i+1].content}; answer: {results1['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    candidate_solutions = []
    for i in range(3):
        cot_instruction2 = f"Subtask 2.{i+1}: Solve the system of equations algebraically or numerically for s and t. Express t in terms of s from the first equation, substitute into the second, and solve for s. Return candidate solution as {'s': value, 't': value} and verify by plugging back into both equations. Iteration {i+1}."
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_2_{i+1}",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, iteration {i+1}, solving for s and t, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2.{i+1} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        candidate_solutions.append(results2['answer'].content)

    cot_sc_instruction3 = "Subtask 3: Using candidate solutions from Subtask 2, generate multiple solution attempts to solve the system and compute the total time in minutes when walking at speed s + 0.5 km/h, including coffee shop time t. Cross-validate and vote on the best solution."
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']] + candidate_solutions,
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, multiple attempts solving and computing total time, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction4 = "Subtask 4: Debate the candidate total times from Subtask 3. Agents argue for or against each candidate based on mathematical consistency and problem constraints to select the most accurate total time."
    final_decision_instruction4 = "Subtask 4: Make final decision on the most accurate total time based on the debate."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query"] + results3['list_thinking'] + results3['list_answer'],
        'input': [taskInfo] + results3['list_answer'],
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
            agents.append(f"Debate agent {agent.id}, round {round}, debating candidate total times, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding most accurate total time, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction5 = "Subtask 5: Review and revise the final total time solution from Subtask 4 for accuracy, consistency, and completeness. Provide detailed feedback and corrections if needed."
    critic_instruction5 = "Please critique the reviewed solution and suggest improvements or confirm correctness."
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc5 = {
        'instruction': critic_instruction5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc5,
        critic_desc=critic_desc5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, reviewing and revising final solution, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correctness: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][i+1].content}; answer: {results5['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_instruction6 = "Subtask 6: Validate and format the final total time in minutes as an integer, ensuring it includes walking time at speed s + 0.5 km/h plus coffee shop time t, and confirm accuracy."
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, validating and formatting final total time, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
