async def forward_163(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = 'Sub-task 1: Extract the orbital periods P1 and P2 (in years) for system_1 and system_2 from the query text.'
    cot_agent1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction1, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1.id}, extracting periods, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction2 = 'Sub-task 2: Extract radial‐velocity semi‐amplitudes for both stars in each system and compute K_sum1 and K_sum2 (in km/s).'
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinking_map2 = {}
    answer_map2 = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction2, 'context': ['user query','thinking1','answer1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents2[i].id}, extracting amplitudes and summing, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers2.append(answer2_i.content)
        thinking_map2[answer2_i.content] = thinking2_i
        answer_map2[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_map2[answer2_content]
    answer2 = answer_map2[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_instruction3 = 'Sub-task 3: Convert P1 and P2 from years to seconds, and convert K_sum1 and K_sum2 from km/s to m/s.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction3, 'context': ['user query','thinking1','answer1','thinking2','answer2'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, converting units, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_instruction4 = 'Sub-task 4: Compute the semi-major axes a1 and a2 for each system using a = (K_sum * P)/(2π), with the converted SI units from subtask_3.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_instruction4, 'context': ['user query','thinking3','answer3'], 'agent_collaboration': 'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, computing semi-major axes, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    cot_instruction5 = 'Sub-task 5: Compute the total masses M1 and M2 of system_1 and system_2 via Kepler’s third-law form M = (4π² a³)/(G P²), using G = 6.674×10⁻¹¹ m³kg⁻¹s⁻².'
    cot_agent5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': cot_instruction5, 'context': ['user query','thinking4','answer4'], 'agent_collaboration': 'CoT'}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent5.id}, computing masses, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    cot_reflect_instruction6 = 'Sub-task 6: Calculate the mass ratio R = M1/M2 and perform a self-consistency check by comparing with the proportional relation R_alt = (K_sum1/K_sum2)^3*(P1/P2), ensuring both agree.'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N6 = self.max_round
    inputs6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking5, answer5]
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot_reflect_instruction6, 'context': ['user query','thinking1','answer1','thinking2','answer2','thinking5','answer5'], 'agent_collaboration': 'Reflexion'}
    thinking6, answer6 = await cot_agent6(inputs6, cot_reflect_instruction6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent6.id}, computing ratios and checks, thinking: {thinking6.content}; answer: {answer6.content}')
    for i in range(N6):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], 'Please review the computed ratios and ensure consistency between R and R_alt.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent6.id}, round {i}, feedback: {feedback6.content}; correct: {correct6.content}')
        if correct6.content == 'True':
            break
        inputs6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent6(inputs6, cot_reflect_instruction6, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent6.id}, refining ratios, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    debate_instruction7 = 'Sub-task 7: Interpret R as the factor by which system_1 is more massive than system_2, verify it matches one of the provided choices, and return the corresponding letter (A–D).'
    debate_agents7 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N7 = self.max_round
    all_thinking7 = [[] for _ in range(N7)]
    all_answer7 = [[] for _ in range(N7)]
    subtask_desc7 = {'subtask_id': 'subtask_7', 'instruction': debate_instruction7, 'context': ['user query','thinking6','answer6'], 'agent_collaboration': 'Debate'}
    for r in range(N7):
        for i, agent in enumerate(debate_agents7):
            if r == 0:
                t7, a7 = await agent([taskInfo, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                t7, a7 = await agent(inputs7, debate_instruction7, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, thinking: {t7.content}; answer: {a7.content}')
            all_thinking7[r].append(t7)
            all_answer7[r].append(a7)
    final_decision7 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision7([taskInfo] + all_thinking7[-1] + all_answer7[-1], 'Sub-task 7: Make final decision on the mass ratio choice.', is_sub_task=True)
    agents.append(f'Final Decision agent, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response'] = {'thinking': thinking7, 'answer': answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs