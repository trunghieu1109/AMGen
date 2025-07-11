async def forward_163(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Extract orbital periods P1 and P2 for system_1 and system_2 from the query.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting periods, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 2: Extract radial-velocity amplitudes K11, K12 for system_1 and K21, K22 for system_2 from the query.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking2, answer2 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting amplitudes, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 3: Compute total RV amplitudes sumK1 = K11 + K12 and sumK2 = K21 + K22.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction, 'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing sums, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    sc_instruction = 'Sub-task 4: Compute mass ratio M1/M2 = [P1 * (sumK1)^3] / [P2 * (sumK2)^3], assuming sin i constant.'
    N = self.max_sc
    sc_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': sc_instruction, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 3', 'answer of subtask 3'], 'agent_collaboration': 'SC_CoT'}
    for agent in sc_agents:
        t, a = await agent([taskInfo, thinking1, answer1, thinking3, answer3], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing ratio, thinking: {t.content}; answer: {a.content}")
        possible_answers.append(a.content)
        thinking_map[a.content] = t
        answer_map[a.content] = a
    chosen = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinking_map[chosen]
    answer4 = answer_map[chosen]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    debate_instruction = 'Sub-task 5: Compare the computed mass ratio to the provided choices and select the closest choice.'
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R = self.max_round
    all_thinking = [[] for _ in range(R)]
    all_answer = [[] for _ in range(R)]
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': debate_instruction, 'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4'], 'agent_collaboration': 'Debate'}
    for r in range(R):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                t5, a5 = await agent([taskInfo, thinking4, answer4], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking4, answer4] + all_thinking[r-1] + all_answer[r-1]
                t5, a5 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t5.content}; answer: {a5.content}")
            all_thinking[r].append(t5)
            all_answer[r].append(a5)
    final_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 'Sub-task 5: Make final decision on the mass ratio choice.', is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs