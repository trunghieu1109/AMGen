async def forward_181(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Extract the Mott–Gurney equation, its variables, and the four candidate statements from the query.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, extracting equation and variables, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    sc_instruction = 'Sub-task 2: List the fundamental assumptions required for the Mott–Gurney equation to hold.'
    N = self.max_sc
    sc_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':sc_instruction,'context':['user query','thinking1','answer1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking2_i, answer2_i = await sc_agents[i]([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {sc_agents[i].id}, listing assumptions, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    instruction3 = 'Sub-task 3: Evaluate whether choice A (trap-free single-carrier device with no carrier injection barrier and negligible diffusion current) matches the Mott–Gurney assumptions.'
    cot3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':instruction3,'context':['user query','thinking2','answer2'],'agent_collaboration':'CoT'}
    thinking3, answer3 = await cot3([taskInfo, thinking2, answer2], instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot3.id}, evaluating Choice 1, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    instruction4 = 'Sub-task 4: Evaluate whether choice B (trap-free single-carrier device with an Ohmic contact and negligible drift current) matches the Mott–Gurney assumptions.'
    cot4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':instruction4,'context':['user query','thinking2','answer2'],'agent_collaboration':'CoT'}
    thinking4, answer4 = await cot4([taskInfo, thinking2, answer2], instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot4.id}, evaluating Choice 2, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    instruction5 = 'Sub-task 5: Evaluate whether choice C (two-carrier device with an Ohmic contact and negligible diffusion current) matches the Mott–Gurney assumptions.'
    cot5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':instruction5,'context':['user query','thinking2','answer2'],'agent_collaboration':'CoT'}
    thinking5, answer5 = await cot5([taskInfo, thinking2, answer2], instruction5, is_sub_task=True)
    agents.append(f'CoT agent {cot5.id}, evaluating Choice 3, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    instruction6 = 'Sub-task 6: Evaluate whether choice D (single-carrier device with a Schottky contact and negligible diffusion current) matches the Mott–Gurney assumptions.'
    cot6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':instruction6,'context':['user query','thinking2','answer2'],'agent_collaboration':'CoT'}
    thinking6, answer6 = await cot6([taskInfo, thinking2, answer2], instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot6.id}, evaluating Choice 4, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    instruction7 = 'Sub-task 7: Compare evaluation results of all choices and determine which choice fully satisfies the Mott–Gurney assumptions.'
    cot7 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc7 = {'subtask_id':'subtask_7','instruction':instruction7,'context':['user query','thinking3','answer3','thinking4','answer4','thinking5','answer5','thinking6','answer6'],'agent_collaboration':'CoT'}
    thinking7, answer7 = await cot7([taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6], instruction7, is_sub_task=True)
    agents.append(f'CoT agent {cot7.id}, comparing evaluation results, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response'] = {'thinking':thinking7,'answer':answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])
    final_instruction = 'Sub-task 8: Format the final answer as the single letter (A, B, C, or D).'
    final_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc8 = {'subtask_id':'subtask_8','instruction':final_instruction,'context':['user query','thinking7','answer7'],'agent_collaboration':'CoT'}
    thinking8, answer8 = await final_agent([taskInfo, thinking7, answer7], final_instruction, is_sub_task=True)
    agents.append(f'CoT agent {final_agent.id}, formatting final answer, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    subtask_desc8['response'] = {'thinking':thinking8,'answer':answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs