async def forward_196(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = 'Sub-task 1: Parse the IR spectral data of Compound X to list characteristic absorption bands and their likely functional group correlations.'
    cot_agent1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction1,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1.id}, analyzing IR bands, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction2 = 'Sub-task 2: Parse the 1H NMR data of Compound X to extract chemical shifts, integrations, multiplicities, and assign preliminary proton environments.'
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinkingmapping2 = {}
    answermapping2 = {}
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_sc_instruction2,'context':['user query'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo], cot_sc_instruction2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents2[i].id}, parsing NMR data, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers2.append(answer2_i.content)
        thinkingmapping2[answer2_i.content] = thinking2_i
        answermapping2[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinkingmapping2[answer2_content]
    answer2 = answermapping2[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_instruction3 = 'Sub-task 3: Identify the presence of a carboxylic acid group by correlating IR broad band at 3400–2500 cm-1 and C=O band at 1720 cm-1 with the 10.5 ppm broad singlet in the 1H NMR.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_instruction3,'context':['user query','thinking of subtask 1','answer of subtask 1','thinking of subtask 2','answer of subtask 2'],'agent_collaboration':'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo,thinking1,answer1,thinking2,answer2], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, identifying carboxylic acid group, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_sc_instruction4 = 'Sub-task 4: Determine the substitution pattern on the aromatic ring by analyzing the two doublets at 8.0 ppm (2H) and 7.2 ppm (2H) in the 1H NMR, consistent with a para-disubstituted benzene.'
    N = self.max_sc
    cot_agents4 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers4 = []
    thinkingmapping4 = {}
    answermapping4 = {}
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':cot_sc_instruction4,'context':['user query','thinking of subtask 2','answer of subtask 2'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo,thinking2,answer2], cot_sc_instruction4, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents4[i].id}, determining substitution pattern, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
        possible_answers4.append(answer4_i.content)
        thinkingmapping4[answer4_i.content] = thinking4_i
        answermapping4[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinkingmapping4[answer4_content]
    answer4 = answermapping4[answer4_content]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    cot_sc_instruction5 = 'Sub-task 5: Characterize the alkyl side chain by interpreting the multiplets at 2.9 ppm (1H) and 1.7 ppm (2H), the doublet at 1.4 ppm (3H), and the triplet at 0.9 ppm (3H) to assign a sec-butyl group.'
    N = self.max_sc
    cot_agents5 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers5 = []
    thinkingmapping5 = {}
    answermapping5 = {}
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':cot_sc_instruction5,'context':['user query','thinking of subtask 2','answer of subtask 2'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking5_i, answer5_i = await cot_agents5[i]([taskInfo,thinking2,answer2], cot_sc_instruction5, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents5[i].id}, characterizing alkyl side chain, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
        possible_answers5.append(answer5_i.content)
        thinkingmapping5[answer5_i.content] = thinking5_i
        answermapping5[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinkingmapping5[answer5_content]
    answer5 = answermapping5[answer5_content]
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    cot_reflect_instruction6 = 'Sub-task 6: Assemble the functional groups into a single structure for Compound X: 4-(sec-butyl)benzoic acid.'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs6 = [taskInfo,thinking3,answer3,thinking4,answer4,thinking5,answer5]
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':cot_reflect_instruction6,'context':['user query','thinking of subtask 3','answer of subtask 3','thinking of subtask 4','answer of subtask 4','thinking of subtask 5','answer of subtask 5'],'agent_collaboration':'Reflexion'}
    thinking6, answer6 = await cot_agent6(cot_inputs6, cot_reflect_instruction6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent6.id}, assembling structure, thinking: {thinking6.content}; answer: {answer6.content}')
    for i in range(N_max):
        feedback6, correct6 = await critic_agent6([taskInfo,thinking6,answer6],'please review the assembled structure and provide its limitations.',i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent6.id}, feedback, thinking: {feedback6.content}; answer: {correct6.content}')
        if correct6.content=='True':
            break
        cot_inputs6.extend([thinking6,answer6,feedback6])
        thinking6, answer6 = await cot_agent6(cot_inputs6, cot_reflect_instruction6,i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent6.id}, refining structure, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    cot_instruction7 = 'Sub-task 7: Recall the chemical transformation effected by red phosphorus and hydroiodic acid on carboxylic acids: reduction of –COOH to –CH3.'
    cot_agent7 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc7 = {'subtask_id':'subtask_7','instruction':cot_instruction7,'context':['user query'],'agent_collaboration':'CoT'}
    thinking7, answer7 = await cot_agent7([taskInfo], cot_instruction7, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent7.id}, recalling reaction, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response'] = {'thinking':thinking7,'answer':answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])
    cot_instruction8 = 'Sub-task 8: Apply the HI/red P reduction to the deduced 4-(sec-butyl)benzoic acid to predict the final product: 1-(sec-butyl)-4-methylbenzene.'
    cot_agent8 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc8 = {'subtask_id':'subtask_8','instruction':cot_instruction8,'context':['user query','thinking of subtask 6','answer of subtask 6','thinking of subtask 7','answer of subtask 7'],'agent_collaboration':'CoT'}
    thinking8, answer8 = await cot_agent8([taskInfo,thinking6,answer6,thinking7,answer7], cot_instruction8, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent8.id}, applying reduction, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    subtask_desc8['response'] = {'thinking':thinking8,'answer':answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])
    debate_instruction9 = 'Sub-task 9: Compare the predicted product structure with the multiple-choice options to find the matching name.'
    debate_agents9 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max9)]
    all_answer9 = [[] for _ in range(N_max9)]
    subtask_desc9 = {'subtask_id':'subtask_9','instruction':debate_instruction9,'context':['user query','thinking of subtask 8','answer of subtask 8'],'agent_collaboration':'Debate'}
    for r in range(N_max9):
        for i,agent in enumerate(debate_agents9):
            if r==0:
                thinking9_i,answer9_i = await agent([taskInfo,thinking8,answer8], debate_instruction9, r, is_sub_task=True)
            else:
                input_infos9 = [taskInfo,thinking8,answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9_i,answer9_i = await agent(input_infos9, debate_instruction9, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, comparing structures, thinking: {thinking9_i.content}; answer: {answer9_i.content}')
            all_thinking9[r].append(thinking9_i)
            all_answer9[r].append(answer9_i)
    final_decision_agent9 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking9,answer9 = await final_decision_agent9([taskInfo] + all_thinking9[-1] + all_answer9[-1], 'Sub-task 9: Make final decision on matching product.', is_sub_task=True)
    agents.append(f'Final Decision agent, deciding match, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    subtask_desc9['response'] = {'thinking':thinking9,'answer':answer9}
    logs.append(subtask_desc9)
    print('Step 9: ', sub_tasks[-1])
    cot_instruction10 = 'Sub-task 10: Select and output the letter corresponding to 1-(sec-butyl)-4-methylbenzene.'
    cot_agent10 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc10 = {'subtask_id':'subtask_10','instruction':cot_instruction10,'context':['user query','thinking of subtask 9','answer of subtask 9'],'agent_collaboration':'CoT'}
    thinking10,answer10 = await cot_agent10([taskInfo,thinking9,answer9], cot_instruction10, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent10.id}, selecting letter, thinking: {thinking10.content}; answer: {answer10.content}')
    sub_tasks.append(f'Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}')
    subtask_desc10['response'] = {'thinking':thinking10,'answer':answer10}
    logs.append(subtask_desc10)
    print('Step 10: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking10,answer10, sub_tasks, agents)
    return final_answer, logs