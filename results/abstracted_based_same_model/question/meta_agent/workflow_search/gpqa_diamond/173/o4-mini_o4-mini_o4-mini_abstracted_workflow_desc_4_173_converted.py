async def forward_173(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = 'Sub-task 1: Compute the rest-mass energies m1 and m2 given m1=2*m2 and m1+m2=0.99*M where M c^2=300 GeV'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, Compute m1 and m2, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2: Compute the total energy released Q = M c^2 - (m1 c^2 + m2 c^2) in GeV'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, Compute Q-value, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    cot_instruction3 = 'Sub-task 3: Compute the classical kinetic energy T1_class = Q * m2 / (m1 + m2)'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction3, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, Compute T1_class, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_instruction4 = 'Sub-task 4: Write the relativistic energy conservation equation sqrt(p^2 + m1^2) + sqrt(p^2 + m2^2) = M c^2 to solve for momentum p'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_instruction4, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2'], 'agent_collaboration': 'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, Write equation for p, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    cot_sc_instruction5 = 'Sub-task 5: Solve the momentum equation sqrt(p^2 + m1^2) + sqrt(p^2 + m2^2) = 300 GeV for p in GeV/c'
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers5 = []
    thinkingmapping5 = {}
    answermapping5 = {}
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': cot_sc_instruction5, 'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N5):
        thinking5, answer5 = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, Solve for p, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers5.append(answer5.content)
        thinkingmapping5[answer5.content] = thinking5
        answermapping5[answer5.content] = answer5
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinkingmapping5[answer5_content]
    answer5 = answermapping5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    cot_instruction6 = 'Sub-task 6: Compute the relativistic kinetic energy T1_rel = sqrt(p^2 + m1^2) - m1 in GeV'
    cot_agent6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot_instruction6, 'context': ['user query', 'thinking of subtask 5', 'answer of subtask 5'], 'agent_collaboration': 'CoT'}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, Compute T1_rel, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])

    cot_instruction7 = 'Sub-task 7: Compute ΔT = T1_rel - T1_class and convert the result into MeV'
    cot_agent7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc7 = {'subtask_id': 'subtask_7', 'instruction': cot_instruction7, 'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3', 'thinking of subtask 6', 'answer of subtask 6'], 'agent_collaboration': 'CoT'}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking3, answer3, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, Compute ΔT, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {'thinking': thinking7, 'answer': answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])

    debate_instruction8 = 'Sub-task 8: Compare the computed ΔT in MeV with choices 10, 5, 2, 20 and select the corresponding letter A–D'
    debate_agents8 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N8 = self.max_round
    all_thinking8 = [[] for _ in range(N8)]
    all_answer8 = [[] for _ in range(N8)]
    subtask_desc8 = {'subtask_id': 'subtask_8', 'instruction': debate_instruction8, 'context': ['user query', 'thinking of subtask 7', 'answer of subtask 7'], 'agent_collaboration': 'Debate'}
    for r in range(N8):
        for i, agent in enumerate(debate_agents8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking7, answer7], debate_instruction8, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(input_infos, debate_instruction8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compare ΔT and select letter, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    final_decision_agent8 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent8([taskInfo] + all_thinking8[-1] + all_answer8[-1], 'Sub-task 8: Make final selection of the letter corresponding to ΔT.', is_sub_task=True)
    agents.append(f"Final Decision agent, select final letter, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {'thinking': thinking8, 'answer': answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs