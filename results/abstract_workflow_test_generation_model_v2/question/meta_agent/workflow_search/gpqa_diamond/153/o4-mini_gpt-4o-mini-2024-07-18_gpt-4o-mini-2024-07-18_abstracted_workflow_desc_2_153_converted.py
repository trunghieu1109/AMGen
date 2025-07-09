async def forward_153(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = 'Sub-task 1: Parse the mass spectrometry data (m/z 156 and 158 peaks) to determine the molecular weight and confirm the presence of a chlorine atom via isotopic pattern.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers1 = []
    thinking_map1 = {}
    answer_map1 = {}
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_sc_instruction, 'context': ['user query'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking1_i, answer1_i = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, parsing mass spec data, thinking: {thinking1_i.content}; answer: {answer1_i.content}')
        possible_answers1.append(answer1_i.content)
        thinking_map1[answer1_i.content] = thinking1_i
        answer_map1[answer1_i.content] = answer1_i
    answer1_content = Counter(possible_answers1).most_common(1)[0][0]
    thinking1 = thinking_map1[answer1_content]
    answer1 = answer_map1[answer1_content]
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_instruction2 = 'Sub-task 2: Interpret the IR spectrum (broad 3500–2700 cm^-1 and sharp 1720 cm^-1 peaks) to identify key functional groups such as carboxylic acid and carbonyl.'
    cot_agent2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_instruction2, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2.id}, interpreting IR spectrum, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    cot_instruction3 = 'Sub-task 3: Analyze the 1H NMR data (11.0 ppm singlet, 8.02 ppm doublet, 7.72 ppm doublet) to assign proton types and deduce substitution pattern on the benzene ring.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction3, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, analyzing NMR data, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_reflect_instruction = 'Sub-task 4: Integrate findings from subtask_1, subtask_2, and subtask_3 to construct a full spectroscopic profile of the unknown compound.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_reflect_instruction, 'context': ['user query','thinking1','answer1','thinking2','answer2','thinking3','answer3'], 'agent_collaboration': 'Reflexion'}
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent4.id}, integrating profile, thinking: {thinking4.content}; answer: {answer4.content}')
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], 'please review the integrated spectroscopic profile and provide its limitations.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent4.id}, providing feedback, thinking: {feedback4.content}; correct: {correct4.content}')
        if correct4.content == 'True':
            break
        cot_inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent4.id}, refining profile, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    debate_instruction5 = 'Sub-task 5: Compare the constructed spectroscopic profile against each multiple-choice candidate structure, noting which features align or conflict.'
    debate_agents5 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': debate_instruction5, 'context': ['user query','thinking4','answer4'], 'agent_collaboration': 'Debate'}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                input_infos5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(input_infos5, debate_instruction5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, comparing candidates, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent5 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make final decision on which candidate best matches the spectroscopic data.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent5.id}, selecting candidate, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    cot_sc_instruction6 = 'Sub-task 6: Rank the four candidates by how well they match the spectroscopic data and select the best structural suggestion.'
    N6 = self.max_sc
    cot_agents6 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_answers6 = []
    thinking_map6 = {}
    answer_map6 = {}
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot_sc_instruction6, 'context': ['user query','thinking5','answer5'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N6):
        thinking6_i, answer6_i = await cot_agents6[i]([taskInfo, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents6[i].id}, ranking candidates, thinking: {thinking6_i.content}; answer: {answer6_i.content}')
        possible_answers6.append(answer6_i.content)
        thinking_map6[answer6_i.content] = thinking6_i
        answer_map6[answer6_i.content] = answer6_i
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinking_map6[answer6_content]
    answer6 = answer_map6[answer6_content]
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs