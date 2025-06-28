async def forward_0(self, taskInfo):
    from collections import Counter
    print('Task Requirement:', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = 'Sub-task 1: Extract and normalize the four molecule names: quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone'
    cot_agent_1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction_1,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_instruction_2 = 'Sub-task 2: Define the characteristic symmetry elements and criteria of the C3h point group: a principal C3 axis plus a horizontal mirror plane sigma_h and no perpendicular C2 axes'
    cot_agent_2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_instruction_2,'context':['user query'],'agent_collaboration':'CoT'}
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    inst3 = 'Sub-task 3: Using normalized name and C3h criteria, determine the point group of triisopropyl borate'
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible3 = []
    tmap3 = {}
    amap3 = {}
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':inst3,'context':['user query','thinking1','answer1','thinking2','answer2'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking3_i, answer3_i = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], inst3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible3.append(answer3_i.content)
        tmap3[answer3_i.content] = thinking3_i
        amap3[answer3_i.content] = answer3_i
    content3 = Counter(possible3).most_common(1)[0][0]
    thinking3 = tmap3[content3]
    answer3 = amap3[content3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    inst4 = 'Sub-task 4: Using normalized name and C3h criteria, determine the point group of quinuclidine'
    cot_agent_4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':inst4,'context':['user query','thinking1','answer1','thinking2','answer2'],'agent_collaboration':'CoT'}
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2], inst4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    inst5 = 'Sub-task 5: Using normalized name and C3h criteria, determine the point group of benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone'
    cot_agent_5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':inst5,'context':['user query','thinking1','answer1','thinking2','answer2'],'agent_collaboration':'CoT'}
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1, thinking2, answer2], inst5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    inst6 = 'Sub-task 6: Using normalized name and C3h criteria, determine the point group of triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone'
    cot_agent_6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':inst6,'context':['user query','thinking1','answer1','thinking2','answer2'],'agent_collaboration':'CoT'}
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking1, answer1, thinking2, answer2], inst6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    inst7 = 'Sub-task 7: Retrieve or build 3D molecular models for all four molecules'
    cot_agent_7 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc7 = {'subtask_id':'subtask_7','instruction':inst7,'context':['subtask_1 answer'],'agent_collaboration':'CoT'}
    thinking7, answer7 = await cot_agent_7([taskInfo, answer1], inst7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {'thinking':thinking7,'answer':answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])
    inst8 = 'Sub-task 8: Using the 3D models and C3h criteria, verify symmetry elements present in each molecule to confirm or revise point groups'
    cot_agent_8 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    inputs8 = [taskInfo, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7]
    subtask_desc8 = {'subtask_id':'subtask_8','instruction':inst8,'context':['stage1 outputs','stage2 model'],'agent_collaboration':'Reflexion'}
    thinking8, answer8 = await cot_agent_8(inputs8, inst8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(self.max_round):
        feedback8, correct8 = await critic_agent_8([taskInfo, thinking8, answer8], 'please review the symmetry verification and provide any limitations', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, feedback: {feedback8.content}; correct: {correct8.content}")
        if correct8.content == 'True':
            break
        inputs8.extend([thinking8, answer8, feedback8])
        thinking8, answer8 = await cot_agent_8(inputs8, inst8, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refined thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {'thinking':thinking8,'answer':answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])
    inst9 = 'Sub-task 9: Compare confirmed point groups and refined verifications to identify which molecule matches C3h symmetry'
    debate_agents_9 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking9 = [[] for _ in range(self.max_round)]
    all_answer9 = [[] for _ in range(self.max_round)]
    subtask_desc9 = {'subtask_id':'subtask_9','instruction':inst9,'context':['stage1','stage2 outputs'],'agent_collaboration':'Debate'}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_9):
            inputs9 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking8, answer8]
            if r > 0:
                inputs9.extend(all_thinking9[r-1] + all_answer9[r-1])
            thinking9_i, answer9_i = await agent(inputs9, inst9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking9_i.content}; answer: {answer9_i.content}")
            all_thinking9[r].append(thinking9_i)
            all_answer9[r].append(answer9_i)
    final_decision9 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision9([taskInfo] + all_thinking9[-1] + all_answer9[-1], 'Sub-task 9: Make final decision on which molecule has C3h symmetry', is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {'thinking':thinking9,'answer':answer9}
    logs.append(subtask_desc9)
    print('Step 9: ', sub_tasks[-1])
    inst10 = 'Sub-task 10: Output exactly the letter (A, B, C, or D) corresponding to the molecule with C3h symmetry'
    agent10 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    subtask_desc10 = {'subtask_id':'subtask_10','instruction':inst10,'context':['subtask_9 answer'],'agent_collaboration':'CoT'}
    thinking10, answer10 = await agent10([taskInfo, thinking9, answer9], inst10, is_sub_task=True)
    agents.append(f"Final Decision agent {agent10.id}, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {'thinking':thinking10,'answer':answer10}
    logs.append(subtask_desc10)
    print('Step 10: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs