async def forward_155(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = 'Sub-task 1: Identify stereochemical epoxide product(s) formed when (E)-oct-4-ene is treated with one equivalent of mCPBA followed by aqueous acid, determining if the epoxide is cis or trans.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['taskInfo'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying stereochemical product for (E)-oct-4-ene, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2: Identify stereochemical epoxide product(s) formed when (Z)-oct-4-ene is treated with one equivalent of mCPBA followed by aqueous acid, determining if the epoxide is cis or trans.'
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['taskInfo', 'thinking1', 'answer1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents2[i].id}, identifying stereochemical product for (Z)-oct-4-ene, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    cot_reflect_instruction = 'Sub-task 3: Determine chirality and number of stereoisomers (enantiomers vs. meso) for the trans-epoxide product from subtask_1.'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs3 = [taskInfo, thinking1, answer1]
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_reflect_instruction, 'context': ['taskInfo', 'thinking1', 'answer1'], 'agent_collaboration': 'Reflexion'}
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent3.id}, determining chirality and stereoisomer count for trans-epoxide, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback_i, correct_i = await critic_agent([taskInfo, thinking3, answer3], 'Please review the chirality and stereoisomer count determination and provide its limitations.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, feedback round {i}, feedback: {feedback_i.content}; correct: {correct_i.content}')
        if correct_i.content == 'True':
            break
        cot_inputs3.extend([thinking3, answer3, feedback_i])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent3.id}, refinement round {i+1}, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_sc_instruction4 = 'Sub-task 4: Determine chirality and number of stereoisomers (enantiomers vs. meso) for the cis-epoxide product from subtask_2.'
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers4 = []
    thinkingmapping4 = {}
    answermapping4 = {}
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_sc_instruction4, 'context': ['taskInfo', 'thinking2', 'answer2'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking2, answer2], cot_sc_instruction4, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents4[i].id}, determining chirality and stereoisomer count for cis-epoxide, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
        possible_answers4.append(answer4_i.content)
        thinkingmapping4[answer4_i.content] = thinking4_i
        answermapping4[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinkingmapping4[answer4_content]
    answer4 = answermapping4[answer4_content]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    debate_instruction5 = 'Sub-task 5: Count the total distinct stereoisomeric species formed from both reactions by summing enantiomeric and meso forms.'
    debate_agents5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': debate_instruction5, 'context': ['taskInfo', 'thinking3', 'answer3', 'thinking4', 'answer4'], 'agent_collaboration': 'Debate'}
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                inputs5 = [taskInfo, thinking3, answer3, thinking4, answer4]
            else:
                inputs5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
            thinking5_i, answer5_i = await agent(inputs5, debate_instruction5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make final decision on total stereoisomeric species count.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent.id}, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    cot_instruction6 = 'Sub-task 6: Predict how many distinct peaks appear on a standard achiral reverse-phase HPLC column, given that enantiomers coelute and diastereomers separate.'
    cot_agent6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot_instruction6, 'context': ['taskInfo', 'thinking5', 'answer5'], 'agent_collaboration': 'CoT'}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, predicting standard HPLC peaks, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])

    cot_instruction7 = 'Sub-task 7: Predict how many distinct peaks appear on a chiral HPLC column, given both enantiomers and diastereomers are resolved.'
    cot_agent7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc7 = {'subtask_id': 'subtask_7', 'instruction': cot_instruction7, 'context': ['taskInfo', 'thinking5', 'answer5'], 'agent_collaboration': 'CoT'}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking5, answer5], cot_instruction7, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent7.id}, predicting chiral HPLC peaks, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response'] = {'thinking': thinking7, 'answer': answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])

    cot_instruction8 = 'Sub-task 8: Match the predicted peak counts on standard and chiral HPLC to the provided choices and select the correct letter (A, B, C, or D).'
    cot_agent8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc8 = {'subtask_id': 'subtask_8', 'instruction': cot_instruction8, 'context': ['taskInfo', 'thinking6', 'answer6', 'thinking7', 'answer7'], 'agent_collaboration': 'CoT'}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking6, answer6, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent8.id}, matching predicted peaks to choices, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    subtask_desc8['response'] = {'thinking': thinking8, 'answer': answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs