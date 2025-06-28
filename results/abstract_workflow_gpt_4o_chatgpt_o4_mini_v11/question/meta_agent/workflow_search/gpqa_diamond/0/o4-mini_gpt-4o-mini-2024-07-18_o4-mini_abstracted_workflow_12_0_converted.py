async def forward_0(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Acquire or generate high-quality 3D geometries for triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone from authoritative sources and output standardized file formats'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, acquiring geometries, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Subtask 1 answer: ', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 2: Reorient each molecule so that principal C3 axis candidates align with laboratory z-axis; output rotated coordinates and record transformation matrices'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_sc_instruction,'context':['user query','response of subtask_1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, reorienting coordinates, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Subtask 2 answer: ', sub_tasks[-1])
    cot_reflect_instruction = 'Sub-task 3: Apply automated symmetry-detection algorithm on reoriented coordinates to produce a table listing all identified symmetry elements with geometric definitions'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_reflect_instruction,'context':['user query','response of subtask_1','response of subtask_2'],'agent_collaboration':'Reflexion'}
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, detecting symmetry elements, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 'Please review the symmetry-detection output and provide limitations', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining symmetry detection, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Subtask 3 answer: ', sub_tasks[-1])
    instruction4 = 'Sub-task 4: Explicitly test for a C3 rotation about the z-axis by rotating structures by ±120° and verifying atomic overlay within tolerance; record pass/fail and deviation metrics'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':instruction4,'context':['user query','response of subtask_3'],'agent_collaboration':'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, testing C3 rotation, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Subtask 4 answer: ', sub_tasks[-1])
    instruction5 = 'Sub-task 5: Test for presence of a horizontal mirror plane perpendicular to z-axis by reflecting across xy-plane and checking positional equivalence within tolerance; record pass/fail and deviation metrics'
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers5 = []
    thinkingmapping5 = {}
    answermapping5 = {}
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':instruction5,'context':['user query','response of subtask_3'],'agent_collaboration':'SC_CoT'}
    for i in range(N5):
        thinking5_i, answer5_i = await cot_agents5[i]([taskInfo, thinking3, answer3], instruction5, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents5[i].id}, testing sigma_h, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
        possible_answers5.append(answer5_i.content)
        thinkingmapping5[answer5_i.content] = thinking5_i
        answermapping5[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinkingmapping5[answer5_content]
    answer5 = answermapping5[answer5_content]
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Subtask 5 answer: ', sub_tasks[-1])
    instruction6 = 'Sub-task 6: Compile results from subtask 4 and 5 into a summary table listing each molecule and boolean flags for C3 and sigma_h plus metrics in JSON'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':instruction6,'context':['user query','response of subtask_4','response of subtask_5'],'agent_collaboration':'CoT'}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking4, answer4, thinking5, answer5], instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, compiling summary, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Subtask 6 answer: ', sub_tasks[-1])
    instruction7 = 'Sub-task 7: From summary table, select molecules with both C3=true and sigma_h=true and list their names and choice numbers'
    cot_agent7 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc7 = {'subtask_id':'subtask_7','instruction':instruction7,'context':['user query','response of subtask_6'],'agent_collaboration':'CoT'}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking6, answer6], instruction7, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent7.id}, selecting candidates, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response'] = {'thinking':thinking7,'answer':answer7}
    logs.append(subtask_desc7)
    print('Subtask 7 answer: ', sub_tasks[-1])
    instruction8 = 'Sub-task 8: Conduct a structured debate on candidate(s): one defends C3h based on metrics, the other raises counter-arguments; adjudicate which claim is stronger'
    debate_agents8 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N8 = self.max_round
    all_thinking8 = [[] for _ in range(N8)]
    all_answer8 = [[] for _ in range(N8)]
    subtask_desc8 = {'subtask_id':'subtask_8','instruction':instruction8,'context':['user query','response of subtask_7'],'agent_collaboration':'Debate'}
    for r in range(N8):
        for i, agent in enumerate(debate_agents8):
            if r == 0:
                inputs8 = [taskInfo, thinking7, answer7]
            else:
                inputs8 = [taskInfo, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
            thinking8_i, answer8_i = await agent(inputs8, instruction8, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, debating candidates, thinking: {thinking8_i.content}; answer: {answer8_i.content}')
            all_thinking8[r].append(thinking8_i)
            all_answer8[r].append(answer8_i)
    final_decision_agent8 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent8([taskInfo] + all_thinking8[-1] + all_answer8[-1], 'Sub-task 8: Adjudicate debate and decide which molecule exhibits C3h symmetry', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent8.id}, adjudicating debate, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    subtask_desc8['response'] = {'thinking':thinking8,'answer':answer8}
    logs.append(subtask_desc8)
    print('Subtask 8 answer: ', sub_tasks[-1])
    instruction9 = 'Sub-task 9: Output only the single letter (A-D) corresponding to the molecule that truly exhibits C3h symmetry'
    final_agent9 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    subtask_desc9 = {'subtask_id':'subtask_9','instruction':instruction9,'context':['user query','response of subtask_8'],'agent_collaboration':'Final'}
    thinking9, answer9 = await final_agent9([taskInfo, answer8], instruction9, is_sub_task=True)
    agents.append(f'Final Decision agent {final_agent9.id}, outputting letter, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    subtask_desc9['response'] = {'thinking':thinking9,'answer':answer9}
    logs.append(subtask_desc9)
    print('Subtask 9 answer: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs