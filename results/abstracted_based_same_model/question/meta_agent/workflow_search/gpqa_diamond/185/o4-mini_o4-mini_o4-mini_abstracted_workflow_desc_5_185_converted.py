async def forward_185(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction0_1 = 'Sub-task 0_1: Construct a clear 2D skeletal diagram of (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene with atom numbering for all ring atoms and the vinyl substituent.'
    cot_agent0_1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc0_1 = {'subtask_id':'subtask_0_1','instruction':cot_instruction0_1,'context':['user query'],'agent_collaboration':'CoT'}
    thinking0_1, answer0_1 = await cot_agent0_1([taskInfo], cot_instruction0_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent0_1.id}, drawing diagram, thinking: {thinking0_1.content}; answer: {answer0_1.content}')
    sub_tasks.append(f'Sub-task 0_1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}')
    print('Step 1: ', sub_tasks[-1])
    subtask_desc0_1['response'] = {'thinking':thinking0_1, 'answer':answer0_1}
    logs.append(subtask_desc0_1)

    cot_sc_instruction0_2 = 'Sub-task 0_2: Using the numbered diagram, identify and record the positions of the π-bonds and the vinyl substituent, and confirm stereochemical configuration at C1 and C4.'
    N = self.max_sc
    cot_agents0_2 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers0_2 = []
    thinking_mapping0_2 = {}
    answer_mapping0_2 = {}
    subtask_desc0_2 = {'subtask_id':'subtask_0_2','instruction':cot_sc_instruction0_2,'context':['user query','thinking0_1','answer0_1'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents0_2:
        thinking, answer = await agent([taskInfo, thinking0_1, answer0_1], cot_sc_instruction0_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {agent.id}, identifying positions, thinking: {thinking.content}; answer: {answer.content}')
        possible_answers0_2.append(answer.content)
        thinking_mapping0_2[answer.content] = thinking
        answer_mapping0_2[answer.content] = answer
    answer0_2_content = Counter(possible_answers0_2).most_common(1)[0][0]
    thinking0_2 = thinking_mapping0_2[answer0_2_content]
    answer0_2 = answer_mapping0_2[answer0_2_content]
    sub_tasks.append(f'Sub-task 0_2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}')
    print('Step 2: ', sub_tasks[-1])
    subtask_desc0_2['response'] = {'thinking':thinking0_2, 'answer':answer0_2}
    logs.append(subtask_desc0_2)

    cot_instruction1_1 = 'Sub-task 1_1: Classify the transformation as a [3,3]-sigmatropic Cope rearrangement and summarize its six-electron cyclic transition state requirements.'
    cot_agent1_1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1_1 = {'subtask_id':'subtask_1_1','instruction':cot_instruction1_1,'context':['user query','thinking0_2','answer0_2'],'agent_collaboration':'CoT'}
    thinking1_1, answer1_1 = await cot_agent1_1([taskInfo, thinking0_2, answer0_2], cot_instruction1_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1_1.id}, classifying reaction, thinking: {thinking1_1.content}; answer: {answer1_1.content}')
    sub_tasks.append(f'Sub-task 1_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}')
    print('Step 3: ', sub_tasks[-1])
    subtask_desc1_1['response'] = {'thinking':thinking1_1, 'answer':answer1_1}
    logs.append(subtask_desc1_1)

    cot_sc_instruction1_2 = 'Sub-task 1_2: List explicitly which bonds break and which form during the [3,3] shift, specifying atom indices from the diagram for each bond.'
    N = self.max_sc
    cot_agents1_2 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers1_2 = []
    thinking_mapping1_2 = {}
    answer_mapping1_2 = {}
    subtask_desc1_2 = {'subtask_id':'subtask_1_2','instruction':cot_sc_instruction1_2,'context':['user query','thinking1_1','answer1_1'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents1_2:
        thinking, answer = await agent([taskInfo, thinking1_1, answer1_1], cot_sc_instruction1_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {agent.id}, listing bond changes, thinking: {thinking.content}; answer: {answer.content}')
        possible_answers1_2.append(answer.content)
        thinking_mapping1_2[answer.content] = thinking
        answer_mapping1_2[answer.content] = answer
    answer1_2_content = Counter(possible_answers1_2).most_common(1)[0][0]
    thinking1_2 = thinking_mapping1_2[answer1_2_content]
    answer1_2 = answer_mapping1_2[answer1_2_content]
    sub_tasks.append(f'Sub-task 1_2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}')
    print('Step 4: ', sub_tasks[-1])
    subtask_desc1_2['response'] = {'thinking':thinking1_2, 'answer':answer1_2}
    logs.append(subtask_desc1_2)

    cot_reflect_instruction1_3 = 'Sub-task 1_3: Validate the proposed bond changes against pericyclic orbital overlap rules and the expected unsaturation pattern of the product.'
    cot_agent1_3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent1_3 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs1_3 = [taskInfo, thinking1_2, answer1_2]
    subtask_desc1_3 = {'subtask_id':'subtask_1_3','instruction':cot_reflect_instruction1_3,'context':['user query','thinking1_2','answer1_2'],'agent_collaboration':'Reflexion'}
    thinking1_3, answer1_3 = await cot_agent1_3(inputs1_3, cot_reflect_instruction1_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent1_3.id}, validating bond changes, thinking: {thinking1_3.content}; answer: {answer1_3.content}')
    for i in range(N_max):
        feedback, correct = await critic_agent1_3([taskInfo, thinking1_2, answer1_2, thinking1_3, answer1_3], 'Please check the validation of bond changes and point out any mismatch with orbital rules or unsaturation pattern.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent1_3.id}, feedback: {feedback.content}; correct: {correct.content}')
        if correct.content == 'True':
            break
        inputs1_3.extend([thinking1_3, answer1_3, feedback])
        thinking1_3, answer1_3 = await cot_agent1_3(inputs1_3, cot_reflect_instruction1_3, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent1_3.id}, refining validation, thinking: {thinking1_3.content}; answer: {answer1_3.content}')
    sub_tasks.append(f'Sub-task 1_3 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}')
    print('Step 5: ', sub_tasks[-1])
    subtask_desc1_3['response'] = {'thinking':thinking1_3, 'answer':answer1_3}
    logs.append(subtask_desc1_3)

    cot_sc_instruction2_1 = 'Sub-task 2_1: Generate three candidate post-rearrangement connectivity scenarios based on the validated bond list.'
    N = self.max_sc
    cot_agents2_1 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2_1 = []
    thinking_list2_1 = []
    answer_list2_1 = []
    subtask_desc2_1 = {'subtask_id':'subtask_2_1','instruction':cot_sc_instruction2_1,'context':['user query','thinking1_3','answer1_3'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents2_1:
        thinking, answer = await agent([taskInfo, thinking1_3, answer1_3], cot_sc_instruction2_1, is_sub_task=True)
        agents.append(f'CoT-SC agent {agent.id}, proposing connectivity scenario, thinking: {thinking.content}; answer: {answer.content}')
        possible_answers2_1.append(answer.content)
        thinking_list2_1.append(thinking)
        answer_list2_1.append(answer)
    sub_tasks.append(f'Sub-task 2_1 output: thinking rounds - {[t.content for t in thinking_list2_1]}; answers - {possible_answers2_1}')
    print('Step 6: ', sub_tasks[-1])
    subtask_desc2_1['response'] = {'thinking':thinking_list2_1, 'answer':answer_list2_1}
    logs.append(subtask_desc2_1)

    cot_instruction2_2 = 'Sub-task 2_2: Perform majority voting among the candidate scenarios to select the most consistent rearranged connectivity.'
    subtask_desc2_2 = {'subtask_id':'subtask_2_2','instruction':cot_instruction2_2,'context':['user query','thinking_list2_1','possible_answers2_1'],'agent_collaboration':'SC_CoT'}
    answer2_2_content = Counter(possible_answers2_1).most_common(1)[0][0]
    idx2_2 = possible_answers2_1.index(answer2_2_content)
    thinking2_2 = thinking_list2_1[idx2_2]
    answer2_2 = answer_list2_1[idx2_2]
    sub_tasks.append(f'Sub-task 2_2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}')
    print('Step 7: ', sub_tasks[-1])
    subtask_desc2_2['response'] = {'thinking':thinking2_2, 'answer':answer2_2}
    logs.append(subtask_desc2_2)

    cot_instruction2_3 = 'Sub-task 2_3: Finalize the validated 2D structure of the rearrangement product with updated atom numbering, ring saturation state, and stereochemistry.'
    cot_agent2_3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc2_3 = {'subtask_id':'subtask_2_3','instruction':cot_instruction2_3,'context':['user query','thinking2_2','answer2_2'],'agent_collaboration':'CoT'}
    thinking2_3, answer2_3 = await cot_agent2_3([taskInfo, thinking2_2, answer2_2], cot_instruction2_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2_3.id}, finalizing structure, thinking: {thinking2_3.content}; answer: {answer2_3.content}')
    sub_tasks.append(f'Sub-task 2_3 output: thinking - {thinking2_3.content}; answer - {answer2_3.content}')
    print('Step 8: ', sub_tasks[-1])
    subtask_desc2_3['response'] = {'thinking':thinking2_3, 'answer':answer2_3}
    logs.append(subtask_desc2_3)

    cot_instruction3_1 = 'Sub-task 3_1: Assign the IUPAC name and generate SMILES or InChI for the finalized product structure.'
    cot_agent3_1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3_1 = {'subtask_id':'subtask_3_1','instruction':cot_instruction3_1,'context':['user query','thinking2_3','answer2_3'],'agent_collaboration':'CoT'}
    thinking3_1, answer3_1 = await cot_agent3_1([taskInfo, thinking2_3, answer2_3], cot_instruction3_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3_1.id}, naming and generating SMILES, thinking: {thinking3_1.content}; answer: {answer3_1.content}')
    sub_tasks.append(f'Sub-task 3_1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}')
    print('Step 9: ', sub_tasks[-1])
    subtask_desc3_1['response'] = {'thinking':thinking3_1, 'answer':answer3_1}
    logs.append(subtask_desc3_1)

    cot_reflect_instruction3_2 = 'Sub-task 3_2: Use a reflexion check to cross-verify that the IUPAC name matches the SMILES/InChI representation, correcting any mismatch.'
    cot_agent3_2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent3_2 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs3_2 = [taskInfo, thinking3_1, answer3_1]
    subtask_desc3_2 = {'subtask_id':'subtask_3_2','instruction':cot_reflect_instruction3_2,'context':['user query','thinking3_1','answer3_1'],'agent_collaboration':'Reflexion'}
    thinking3_2, answer3_2 = await cot_agent3_2(inputs3_2, cot_reflect_instruction3_2, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent3_2.id}, checking name vs SMILES, thinking: {thinking3_2.content}; answer: {answer3_2.content}')
    for i in range(N_max):
        feedback, correct = await critic_agent3_2([taskInfo, thinking3_1, answer3_1, thinking3_2, answer3_2], 'Please verify if the IUPAC name correctly matches the SMILES/InChI and point out mismatches.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent3_2.id}, feedback: {feedback.content}; correct: {correct.content}')
        if correct.content == 'True':
            break
        inputs3_2.extend([thinking3_2, answer3_2, feedback])
        thinking3_2, answer3_2 = await cot_agent3_2(inputs3_2, cot_reflect_instruction3_2, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent3_2.id}, refining verification, thinking: {thinking3_2.content}; answer: {answer3_2.content}')
    sub_tasks.append(f'Sub-task 3_2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}')
    print('Step 10: ', sub_tasks[-1])
    subtask_desc3_2['response'] = {'thinking':thinking3_2, 'answer':answer3_2}
    logs.append(subtask_desc3_2)

    cot_instruction3_3 = 'Sub-task 3_3: Convert each of the four provided choice names into SMILES or InChI for structural comparison.'
    cot_agent3_3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5)
    subtask_desc3_3 = {'subtask_id':'subtask_3_3','instruction':cot_instruction3_3,'context':['user query','thinking3_2','answer3_2'],'agent_collaboration':'CoT'}
    thinking3_3, answer3_3 = await cot_agent3_3([taskInfo, thinking3_2, answer3_2], cot_instruction3_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3_3.id}, converting choices to SMILES, thinking: {thinking3_3.content}; answer: {answer3_3.content}')
    sub_tasks.append(f'Sub-task 3_3 output: thinking - {thinking3_3.content}; answer - {answer3_3.content}')
    print('Step 11: ', sub_tasks[-1])
    subtask_desc3_3['response'] = {'thinking':thinking3_3, 'answer':answer3_3}
    logs.append(subtask_desc3_3)

    debate_instruction3_4 = 'Sub-task 3_4: Compare the validated product’s structure and naming features (unsaturation positions, stereochemistry) against each choice and select the matching letter (A–D).'
    debate_agents3_4 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking3_4 = [[] for _ in range(N_max)]
    all_answer3_4 = [[] for _ in range(N_max)]
    subtask_desc3_4 = {'subtask_id':'subtask_3_4','instruction':debate_instruction3_4,'context':['user query','thinking3_3','answer3_3'],'agent_collaboration':'Debate'}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents3_4):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3_3, answer3_3], debate_instruction3_4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3_3, answer3_3] + all_thinking3_4[r-1] + all_answer3_4[r-1]
                thinking, answer = await agent(inputs, debate_instruction3_4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, comparing choices, thinking: {thinking.content}; answer: {answer.content}')
            all_thinking3_4[r].append(thinking)
            all_answer3_4[r].append(answer)
    final_decision3_4 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking3_4, answer3_4 = await final_decision3_4([taskInfo] + all_thinking3_4[-1] + all_answer3_4[-1], 'Sub-task 3_4: Select the letter corresponding to the matching choice.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision3_4.id}, selecting choice, thinking: {thinking3_4.content}; answer: {answer3_4.content}')
    sub_tasks.append(f'Sub-task 3_4 output: thinking - {thinking3_4.content}; answer - {answer3_4.content}')
    print('Step 12: ', sub_tasks[-1])
    subtask_desc3_4['response'] = {'thinking':thinking3_4, 'answer':answer3_4}
    logs.append(subtask_desc3_4)

    final_answer = await self.make_final_answer(thinking3_4, answer3_4, sub_tasks, agents)
    return final_answer, logs