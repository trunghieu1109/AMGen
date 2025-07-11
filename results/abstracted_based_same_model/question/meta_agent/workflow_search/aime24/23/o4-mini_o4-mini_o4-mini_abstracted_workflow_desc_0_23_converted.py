async def forward_23(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = 'Sub-task 1: Define symbolic variables d_{i,j} for each cell in the 2×3 grid, where i∈{1,2} and j∈{1,2,3}.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, define variables, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2: Formulate the row-sum constraint: the number (d_{1,1}d_{1,2}d_{1,3}) + (d_{2,1}d_{2,2}d_{2,3}) = 999.'
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible2 = []
    map2_t = {}
    map2_a = {}
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_sc_instruction,'context':['user query','thinking1','answer1'],'agent_collaboration':'SC_CoT'}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents2[i].id}, formulate row-sum, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible2.append(answer2_i.content)
        map2_t[answer2_i.content] = thinking2_i
        map2_a[answer2_i.content] = answer2_i
    answer2_content = Counter(possible2).most_common(1)[0][0]
    thinking2 = map2_t[answer2_content]
    answer2 = map2_a[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    cot_instruction3 = 'Sub-task 3: Formulate the column-sum constraint: sum of 10*d_{1,j}+d_{2,j} for j=1,2,3 equals 99.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_instruction3,'context':['user query','thinking1','answer1'],'agent_collaboration':'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, formulate column-sum, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_instruction4 = 'Sub-task 4: State the domain constraint: each d_{i,j} is an integer digit from 0 to 9.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':cot_instruction4,'context':['user query'],'agent_collaboration':'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, state domain constraint, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    cot_instruction5 = 'Sub-task 5: Enumerate all top-row numbers 000–999 as candidates for row-sum constraint.'
    cot_agent5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':cot_instruction5,'context':['user query','thinking2','answer2','thinking4','answer4'],'agent_collaboration':'CoT'}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking2, answer2, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent5.id}, enumerate top-row candidates, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    cot_instruction6 = 'Sub-task 6: Compute bottom-row digits for each candidate by subtracting top-row from 999 and extracting digits.'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':cot_instruction6,'context':['user query','thinking5','answer5'],'agent_collaboration':'CoT'}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, compute bottom-row digits, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])

    cot_sc_instruction7 = 'Sub-task 7: Validate bottom-row domain: ensure each digit is between 0 and 9.'
    N7 = self.max_sc
    sc_agents7 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N7)]
    possible7 = []
    map7_t = {}
    map7_a = {}
    subtask_desc7 = {'subtask_id':'subtask_7','instruction':cot_sc_instruction7,'context':['user query','thinking6','answer6','thinking4','answer4'],'agent_collaboration':'SC_CoT'}
    for i in range(N7):
        thinking7_i, answer7_i = await sc_agents7[i]([taskInfo, thinking6, answer6, thinking4, answer4], cot_sc_instruction7, is_sub_task=True)
        agents.append(f'CoT-SC agent {sc_agents7[i].id}, validate domain constraint, thinking: {thinking7_i.content}; answer: {answer7_i.content}')
        possible7.append(answer7_i.content)
        map7_t[answer7_i.content] = thinking7_i
        map7_a[answer7_i.content] = answer7_i
    answer7_content = Counter(possible7).most_common(1)[0][0]
    thinking7 = map7_t[answer7_content]
    answer7 = map7_a[answer7_content]
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response'] = {'thinking':thinking7,'answer':answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])

    cot_instruction8 = 'Sub-task 8: Validate column-sum for each candidate: compute 10*d_{1,j}+d_{2,j} for j=1,2,3 and check sum equals 99.'
    cot_agent8 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc8 = {'subtask_id':'subtask_8','instruction':cot_instruction8,'context':['user query','thinking6','answer6','thinking3','answer3'],'agent_collaboration':'CoT'}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking6, answer6, thinking3, answer3], cot_instruction8, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent8.id}, validate column-sum constraint, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    subtask_desc8['response'] = {'thinking':thinking8,'answer':answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])

    cot_reflect_instruction9 = 'Sub-task 9: Reflexively review each candidate from Sub-task 6 and confirm it satisfies all constraints; collect valid arrangements.'
    cot_agent9 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent9 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N9 = self.max_round
    cot_inputs9 = [taskInfo, thinking6, answer6, thinking7, answer7, thinking8, answer8]
    subtask_desc9 = {'subtask_id':'subtask_9','instruction':cot_reflect_instruction9,'context':['user query','thinking6','answer6','thinking7','answer7','thinking8','answer8'],'agent_collaboration':'Reflexion'}
    thinking9, answer9 = await cot_agent9(cot_inputs9, cot_reflect_instruction9, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent9.id}, initial review, thinking: {thinking9.content}; answer: {answer9.content}')
    for i in range(N9):
        feedback9, correct9 = await critic_agent9([taskInfo, thinking9, answer9], 'Please review the filtering of candidates and provide any limitations.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent9.id}, feedback, thinking: {feedback9.content}; answer: {correct9.content}')
        if correct9.content == 'True':
            break
        cot_inputs9.extend([thinking9, answer9, feedback9])
        thinking9, answer9 = await cot_agent9(cot_inputs9, cot_reflect_instruction9, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent9.id}, refinement, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    subtask_desc9['response'] = {'thinking':thinking9,'answer':answer9}
    logs.append(subtask_desc9)
    print('Step 9: ', sub_tasks[-1])

    cot_instruction10 = 'Sub-task 10: Count the total number of valid grid assignments collected in Sub-task 9.'
    cot_agent10 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc10 = {'subtask_id':'subtask_10','instruction':cot_instruction10,'context':['user query','thinking9','answer9'],'agent_collaboration':'CoT'}
    thinking10, answer10 = await cot_agent10([taskInfo, thinking9, answer9], cot_instruction10, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent10.id}, count valid assignments, thinking: {thinking10.content}; answer: {answer10.content}')
    sub_tasks.append(f'Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}')
    subtask_desc10['response'] = {'thinking':thinking10,'answer':answer10}
    logs.append(subtask_desc10)
    print('Step 10: ', sub_tasks[-1])

    debate_instruction11 = 'Sub-task 11: Return the integer count as the final answer.'
    debate_agents11 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking11 = []
    all_answer11 = []
    subtask_desc11 = {'subtask_id':'subtask_11','instruction':debate_instruction11,'context':['user query','thinking10','answer10'],'agent_collaboration':'Debate'}
    for agent in debate_agents11:
        thinking11_i, answer11_i = await agent([taskInfo, thinking10, answer10], debate_instruction11, is_sub_task=True)
        agents.append(f'Debate agent {agent.id}, propose final count, thinking: {thinking11_i.content}; answer: {answer11_i.content}')
        all_thinking11.append(thinking11_i)
        all_answer11.append(answer11_i)
    final_decision_agent11 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking11, answer11 = await final_decision_agent11([taskInfo] + all_thinking11 + all_answer11, debate_instruction11, is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent11.id}, decide final count, thinking: {thinking11.content}; answer: {answer11.content}')
    sub_tasks.append(f'Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}')
    subtask_desc11['response'] = {'thinking':thinking11,'answer':answer11}
    logs.append(subtask_desc11)
    print('Step 11: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer, logs