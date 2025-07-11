async def forward_182(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = 'Sub-task 1: Parse the IUPAC name "2-formyl-5-vinylcyclohex-3-enecarboxylic acid" to draw the skeletal structure and list all functional groups (aldehyde, carboxylic acid, vinyl, cyclohexene double bond).'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1,answer1 = await cot_agent([taskInfo],cot_instruction,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, parsing IUPAC name, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response']={'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2a: Predict the chemical transformation of the aldehyde (formyl) group under Red P and excess HI, including mechanistic justification.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc2a = {'subtask_id':'subtask_2a','instruction':cot_sc_instruction,'context':['user query','thinking1','answer1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        t,a = await cot_agents[i]([taskInfo,thinking1,answer1],cot_sc_instruction,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, predicting aldehyde transformation, thinking: {t.content}; answer: {a.content}')
        possible_answers.append(a.content)
        thinking_mapping[a.content] = t
        answer_mapping[a.content] = a
    answer2a_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2a = thinking_mapping[answer2a_content]
    answer2a = answer_mapping[answer2a_content]
    sub_tasks.append(f'Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}')
    subtask_desc2a['response']={'thinking':thinking2a,'answer':answer2a}
    logs.append(subtask_desc2a)
    print('Step 2a: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2b: Predict the fate of the carboxylic acid group under Red P and excess HI (reduction to alcohol, decarboxylation, or deoxygenation) with mechanistic support.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc2b = {'subtask_id':'subtask_2b','instruction':cot_sc_instruction,'context':['user query','thinking1','answer1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        t,a = await cot_agents[i]([taskInfo,thinking1,answer1],cot_sc_instruction,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, predicting acid fate, thinking: {t.content}; answer: {a.content}')
        possible_answers.append(a.content)
        thinking_mapping[a.content] = t
        answer_mapping[a.content] = a
    answer2b_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2b = thinking_mapping[answer2b_content]
    answer2b = answer_mapping[answer2b_content]
    sub_tasks.append(f'Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}')
    subtask_desc2b['response']={'thinking':thinking2b,'answer':answer2b}
    logs.append(subtask_desc2b)
    print('Step 2b: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2c: Determine the reactivity of each C=C bond (vinyl side chain and cyclohexene double bond) under Red P and excess HI, specifying whether they are reduced or retained.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc2c = {'subtask_id':'subtask_2c','instruction':cot_sc_instruction,'context':['user query','thinking1','answer1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        t,a = await cot_agents[i]([taskInfo,thinking1,answer1],cot_sc_instruction,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, analyzing C=C reactivity, thinking: {t.content}; answer: {a.content}')
        possible_answers.append(a.content)
        thinking_mapping[a.content] = t
        answer_mapping[a.content] = a
    answer2c_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2c = thinking_mapping[answer2c_content]
    answer2c = answer_mapping[answer2c_content]
    sub_tasks.append(f'Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}')
    subtask_desc2c['response']={'thinking':thinking2c,'answer':answer2c}
    logs.append(subtask_desc2c)
    print('Step 2c: ', sub_tasks[-1])

    cot_reflect_instruction = 'Sub-task 3: Critically evaluate and validate the predictions from subtasks 2a–2c against reference Red P/HI reactivity to identify and correct any flawed assumptions.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_reflect_instruction,'context':['user query','thinking2a','answer2a','thinking2b','answer2b','thinking2c','answer2c'],'agent_collaboration':'Reflexion'}
    cot_inputs = [taskInfo,thinking2a,answer2a,thinking2b,answer2b,thinking2c,answer2c]
    thinking3,answer3 = await cot_agent(cot_inputs,cot_reflect_instruction,0,is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, validating predictions, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback,correct = await critic_agent([taskInfo,thinking3,answer3],'please review the validation of predictions and provide corrections.',i,is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3,answer3,feedback])
        thinking3,answer3 = await cot_agent(cot_inputs,cot_reflect_instruction,i+1,is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining validation, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response']={'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 4: Integrate the validated outcomes of subtasks 2a–2c into a single coherent final product skeletal structure.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':cot_instruction,'context':['user query','thinking3','answer3'],'agent_collaboration':'CoT'}
    thinking4,answer4 = await cot_agent([taskInfo,thinking3,answer3],cot_instruction,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, constructing final structure, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response']={'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 5: Derive the molecular formula of the predicted product by counting all atoms (C, H, O, etc.) in the structure from subtask 4.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':cot_instruction,'context':['user query','thinking4','answer4'],'agent_collaboration':'CoT'}
    thinking5,answer5 = await cot_agent([taskInfo,thinking4,answer4],cot_instruction,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, deriving molecular formula, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response']={'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 6: Calculate the index of hydrogen deficiency (IHD) for the product using the formula IHD = (2C + 2 – H)/2 based on the molecular formula.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':cot_sc_instruction,'context':['user query','thinking5','answer5'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        t,a = await cot_agents[i]([taskInfo,thinking5,answer5],cot_sc_instruction,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, calculating IHD, thinking: {t.content}; answer: {a.content}')
        possible_answers.append(a.content)
        thinking_mapping[a.content] = t
        answer_mapping[a.content] = a
    answer6_content = Counter(possible_answers).most_common(1)[0][0]
    thinking6 = thinking_mapping[answer6_content]
    answer6 = answer_mapping[answer6_content]
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response']={'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])

    cot_reflect_instruction = 'Sub-task 7: Perform a final consistency check by comparing the structural reasoning, molecular formula, and IHD calculation to ensure no discrepancies.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc7 = {'subtask_id':'subtask_7','instruction':cot_reflect_instruction,'context':['user query','thinking4','answer4','thinking6','answer6'],'agent_collaboration':'Reflexion'}
    cot_inputs = [taskInfo,thinking4,answer4,thinking6,answer6]
    thinking7,answer7 = await cot_agent(cot_inputs,cot_reflect_instruction,0,is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, performing consistency check, thinking: {thinking7.content}; answer: {answer7.content}')
    for i in range(N_max):
        feedback,correct = await critic_agent([taskInfo,thinking7,answer7],'please review the consistency check and identify discrepancies.',i,is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking7,answer7,feedback])
        thinking7,answer7 = await cot_agent(cot_inputs,cot_reflect_instruction,i+1,is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining consistency check, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response']={'thinking':thinking7,'answer':answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])

    debate_instruction = 'Sub-task 8: Match the verified IHD value to the provided choices (A:1, B:3, C:0, D:5) and select the correct letter.'
    debate_agents = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc8 = {'subtask_id':'subtask_8','instruction':debate_instruction,'context':['user query','thinking6','answer6'],'agent_collaboration':'Debate'}
    for r in range(N_max):
        for i,agent in enumerate(debate_agents):
            if r == 0:
                t,a = await agent([taskInfo,thinking6,answer6],debate_instruction,r,is_sub_task=True)
            else:
                inputs = [taskInfo,thinking6,answer6] + all_thinking[r-1] + all_answer[r-1]
                t,a = await agent(inputs,debate_instruction,r,is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, selecting choice, thinking: {t.content}; answer: {a.content}')
            all_thinking[r].append(t)
            all_answer[r].append(a)
    final_decision_agent = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking8,answer8 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 'Sub-task 8: Make final decision on the correct choice letter.', is_sub_task=True)
    agents.append(f'Final Decision agent, selecting choice, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    subtask_desc8['response']={'thinking':thinking8,'answer':answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8,answer8,sub_tasks,agents)
    return final_answer,logs