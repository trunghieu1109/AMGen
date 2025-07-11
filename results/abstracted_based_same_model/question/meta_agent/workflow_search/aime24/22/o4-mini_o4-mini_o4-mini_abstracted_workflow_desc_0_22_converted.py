async def forward_22(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Analyze the definition of the median and deduce whether the list length n must be odd or even given the median is an integer not present in the list'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1', 'instruction':cot_instruction, 'context':['user query'], 'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing definition of median, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print('Step 1: ', sub_tasks[-1])
    subtask_desc1['response'] = {'thinking':thinking1, 'answer':answer1}
    logs.append(subtask_desc1)
    sc_instruction2 = 'Sub-task 2: Conclude that the list length n must be even and determine the general form n=2k under sum=30 and mode constraints'
    N2 = self.max_sc
    sc_agents2 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinking_mapping2 = {}
    answer_mapping2 = {}
    subtask_desc2 = {'subtask_id':'subtask_2', 'instruction':sc_instruction2, 'context':['user query','thinking of subtask 1','answer of subtask 1'], 'agent_collaboration':'SC_CoT'}
    for i in range(N2):
        thinking2_i, answer2_i = await sc_agents2[i]([taskInfo, thinking1, answer1], sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents2[i].id}, concluding list length and form, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinking_mapping2[answer2_i.content] = thinking2_i
        answer_mapping2[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_mapping2[answer2_content]
    answer2 = answer_mapping2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print('Step 2: ', sub_tasks[-1])
    subtask_desc2['response'] = {'thinking':thinking2, 'answer':answer2}
    logs.append(subtask_desc2)
    cot3_instruction = 'Sub-task 3: Determine possible frequency f9 of the mode value 9 given sum=30 and that 9 appears strictly more than any other integer'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id':'subtask_3', 'instruction':cot3_instruction, 'context':['user query','thinking of subtask 2','answer of subtask 2'], 'agent_collaboration':'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot3_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, determining f9 possibilities, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print('Step 3: ', sub_tasks[-1])
    subtask_desc3['response'] = {'thinking':thinking3, 'answer':answer3}
    logs.append(subtask_desc3)
    ref4_instruction = 'Sub-task 4: For each possible f9, compute the remaining sum and r=2k-f9, enforce r>=2 and even, then generate all multisets of r positive integers (none equal to 9) summing to that remainder'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N4 = self.max_round
    inputs4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {'subtask_id':'subtask_4', 'instruction':ref4_instruction, 'context':['user query','thinking of subtask 2','answer of subtask 2','thinking of subtask 3','answer of subtask 3'], 'agent_collaboration':'Reflexion'}
    thinking4, answer4 = await cot_agent4(inputs4, ref4_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, generating multisets, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N4):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], 'Please review the multiset generation and provide limitations.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback on multiset generation, thinking: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == 'True':
            break
        inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(inputs4, ref4_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refining multisets, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print('Step 4: ', sub_tasks[-1])
    subtask_desc4['response'] = {'thinking':thinking4, 'answer':answer4}
    logs.append(subtask_desc4)
    sc5_instruction = 'Sub-task 5: For each multiset, add f9 copies of 9, sort the full list, and compute its median m=(a_k + a_{k+1})/2'
    N5 = self.max_sc
    sc_agents5 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers5 = []
    thinking_map5 = {}
    answer_map5 = {}
    subtask_desc5 = {'subtask_id':'subtask_5', 'instruction':sc5_instruction, 'context':['user query','thinking of subtask 4','answer of subtask 4'], 'agent_collaboration':'SC_CoT'}
    for i in range(N5):
        thinking5_i, answer5_i = await sc_agents5[i]([taskInfo, thinking4, answer4], sc5_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents5[i].id}, computing median for each list, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_answers5.append(answer5_i.content)
        thinking_map5[answer5_i.content] = thinking5_i
        answer_map5[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinking_map5[answer5_content]
    answer5 = answer_map5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print('Step 5: ', sub_tasks[-1])
    subtask_desc5['response'] = {'thinking':thinking5, 'answer':answer5}
    logs.append(subtask_desc5)
    cot6_instruction = 'Sub-task 6: Filter the lists by requiring that the median is an integer not present among the list entries'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id':'subtask_6', 'instruction':cot6_instruction, 'context':['user query','thinking of subtask 5','answer of subtask 5'], 'agent_collaboration':'CoT'}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot6_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, filtering by median condition, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print('Step 6: ', sub_tasks[-1])
    subtask_desc6['response'] = {'thinking':thinking6, 'answer':answer6}
    logs.append(subtask_desc6)
    ref7_instruction = 'Sub-task 7: Verify among the filtered lists that 9 remains the unique mode'
    cot_agent7 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent7 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N7 = self.max_round
    inputs7 = [taskInfo, thinking5, answer5, thinking6, answer6]
    subtask_desc7 = {'subtask_id':'subtask_7', 'instruction':ref7_instruction, 'context':['user query','thinking of subtask 5','answer of subtask 5','thinking of subtask 6','answer of subtask 6'], 'agent_collaboration':'Reflexion'}
    thinking7, answer7 = await cot_agent7(inputs7, ref7_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent7.id}, verifying unique mode, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N7):
        feedback7, correct7 = await critic_agent7([taskInfo, thinking7, answer7], 'Please review unique mode verification and provide limitations.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent7.id}, feedback on unique mode, thinking: {feedback7.content}; correct: {correct7.content}")
        if correct7.content == 'True':
            break
        inputs7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent7(inputs7, ref7_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent7.id}, refining unique mode verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print('Step 7: ', sub_tasks[-1])
    subtask_desc7['response'] = {'thinking':thinking7, 'answer':answer7}
    logs.append(subtask_desc7)
    sc8_instruction = 'Sub-task 8: For each list satisfying all conditions, compute the sum of squares of its elements'
    N8 = self.max_sc
    sc_agents8 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N8)]
    possible_answers8 = []
    thinking_map8 = {}
    answer_map8 = {}
    subtask_desc8 = {'subtask_id':'subtask_8', 'instruction':sc8_instruction, 'context':['user query','thinking of subtask 7','answer of subtask 7'], 'agent_collaboration':'SC_CoT'}
    for i in range(N8):
        thinking8_i, answer8_i = await sc_agents8[i]([taskInfo, thinking7, answer7], sc8_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents8[i].id}, computing sum of squares, thinking: {thinking8_i.content}; answer: {answer8_i.content}")
        possible_answers8.append(answer8_i.content)
        thinking_map8[answer8_i.content] = thinking8_i
        answer_map8[answer8_i.content] = answer8_i
    answer8_content = Counter(possible_answers8).most_common(1)[0][0]
    thinking8 = thinking_map8[answer8_content]
    answer8 = answer_map8[answer8_content]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print('Step 8: ', sub_tasks[-1])
    subtask_desc8['response'] = {'thinking':thinking8, 'answer':answer8}
    logs.append(subtask_desc8)
    debate9_instruction = 'Sub-task 9: Identify the valid list(s) and return the integer value of the sum of squares'
    debate_agents9 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N9 = self.max_round
    all_thinking9 = [[] for _ in range(N9)]
    all_answer9 = [[] for _ in range(N9)]
    subtask_desc9 = {'subtask_id':'subtask_9', 'instruction':debate9_instruction, 'context':['user query','thinking of subtask 8','answer of subtask 8'], 'agent_collaboration':'Debate'}
    for r in range(N9):
        for i, agent in enumerate(debate_agents9):
            if r == 0:
                thinking9_i, answer9_i = await agent([taskInfo, thinking8, answer8], debate9_instruction, r, is_sub_task=True)
            else:
                inputs9 = [taskInfo, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9_i, answer9_i = await agent(inputs9, debate9_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, identifying valid lists and sum of squares, thinking: {thinking9_i.content}; answer: {answer9_i.content}")
            all_thinking9[r].append(thinking9_i)
            all_answer9[r].append(answer9_i)
    final_decision_agent9 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent9([taskInfo] + all_thinking9[-1] + all_answer9[-1], 'Sub-task 9: Make final decision on sum of squares value.', is_sub_task=True)
    agents.append(f"Final Decision agent, determining final sum of squares, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print('Step 9: ', sub_tasks[-1])
    subtask_desc9['response'] = {'thinking':thinking9, 'answer':answer9}
    logs.append(subtask_desc9)
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs