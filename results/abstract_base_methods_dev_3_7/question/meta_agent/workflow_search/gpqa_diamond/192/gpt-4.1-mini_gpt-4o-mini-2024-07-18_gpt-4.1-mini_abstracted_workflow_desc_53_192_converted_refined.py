async def forward_192(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = "Subtask 1: Express the relation between parallax (plx) and distance (r). Show only that r = 1/plx and nothing more. Then validate this expression against the original question and check the mathematical transformation for accuracy."
    critic_instruction1 = "Please review the expression r = 1/plx and provide feedback on its correctness and any missing considerations."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query']
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id='subtask_1',
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, expressing r=1/plx and validating, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback round {i}, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining expression round {i}, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")

    cot_reflect_instruction2 = "Subtask 2: Given N(plx) ∝ 1/plx^5 and plx = 1/r, derive N(r) per unit distance r. Show explicitly that N(r) = N(plx(r)) × |d(plx)/d(r)| and compute the exponent. Then reflect on whether the result is physically consistent with star counts scaling with volume (~ r^3)."
    critic_instruction2 = "Please review the derivation of N(r) including the Jacobian factor and provide feedback on correctness and physical consistency."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id='subtask_2',
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, deriving N(r) with Jacobian and reflecting, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback round {i}, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining derivation round {i}, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")

    debate_instruction3 = "Subtask 3: Debate the correct expression for the variation of number of stars with distance r. Agent A: Substitute plx=1/r directly into N ∝ 1/plx^5 and state the result. Agent B: Argue that densities per unit variable require applying the Jacobian |d(plx)/d(r)| = 1/r^2 and derive the correct exponent."
    final_decision_instruction3 = "Subtask 3: Make final decision on the correct variation of number of stars with distance r, choosing among the given options."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ['user query', results2['thinking'], results2['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    final_decision_desc3 = {
        'instruction': final_decision_instruction3,
        'output': ['thinking', 'answer'],
        'temperature': 0.0
    }
    results3 = await self.debate(
        subtask_id='subtask_3',
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating star count variation, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding correct star count variation, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs
