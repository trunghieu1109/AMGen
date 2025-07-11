async def forward_155(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Determine products and stereochemical outcome for (E)-oct-4-ene with one equivalent of mCPBA followed by aqueous acid.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, determining products for (E)-oct-4-ene, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_instruction2 = 'Sub-task 2: Determine products and stereochemical outcome for (Z)-oct-4-ene with one equivalent of mCPBA followed by aqueous acid.'
    cot_agent2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_instruction2,'context':['user query'],'agent_collaboration':'CoT'}
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2.id}, determining products for (Z)-oct-4-ene, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    sc_instruction = 'Sub-task 3: Classify each epoxide product from subtask 1 and subtask 2 as either an achiral meso compound or an enantiomeric pair.'
    N = self.max_sc
    cot_agents3 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':sc_instruction,'context':['user query','thinking1','answer1','thinking2','answer2'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo,thinking1,answer1,thinking2,answer2], sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents3[i].id}, classifying epoxide chirality, thinking: {thinking3_i.content}; answer: {answer3_i.content}')
        possible_answers.append(answer3_i.content)
        thinking_map[answer3_i.content] = thinking3_i
        answer_map[answer3_i.content] = answer3_i
    answer3_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinking_map[answer3_content]
    answer3 = answer_map[answer3_content]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_instruction4 = 'Sub-task 4: Predict number of distinct peaks on a standard reverse-phase HPLC, accounting for co-elution of enantiomers.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':cot_instruction4,'context':['user query','thinking3','answer3'],'agent_collaboration':'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo,thinking3,answer3], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, predicting standard HPLC peaks, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    cot_instruction5 = 'Sub-task 5: Predict number of distinct peaks on a chiral HPLC column, considering separation of enantiomers and the meso compound.'
    cot_agent5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':cot_instruction5,'context':['user query','thinking3','answer3'],'agent_collaboration':'CoT'}
    thinking5, answer5 = await cot_agent5([taskInfo,thinking3,answer3], cot_instruction5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent5.id}, predicting chiral HPLC peaks, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    debate_instruction = 'Sub-task 6: Match predicted peak counts from subtask 4 and subtask 5 to the provided multiple-choice options (A–D).'
    debate_agents = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':debate_instruction,'context':['user query','thinking4','answer4','thinking5','answer5'],'agent_collaboration':'Debate'}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r==0:
                thinking6, answer6 = await agent([taskInfo,thinking4,answer4,thinking5,answer5], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo,thinking4,answer4,thinking5,answer5] + all_thinking[r-1] + all_answer[r-1]
                thinking6, answer6 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, matching peaks to choices, thinking: {thinking6.content}; answer: {answer6.content}')
            all_thinking[r].append(thinking6)
            all_answer[r].append(answer6)
    final_decision_agent = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], debate_instruction, is_sub_task=True)
    agents.append(f'Final Decision agent, matching predicted peaks to choices, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs