async def forward_154(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = 'Sub-task 1: Extract the P_z operator matrix and the given state vector components from the problem statement in a form suitable for computation.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, extracting P_z matrix and state vector, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2: Verify that the provided state vector is normalized (i.e. ⟨ψ|ψ⟩ = 1) and renormalize if necessary using the extracted state vector.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['user query','thinking of subtask 1','answer of subtask 1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, verifying normalization, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
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

    cot_instruction3 = 'Sub-task 3: Compute the expectation value ⟨P_z⟩ = ψ† P_z ψ using the extracted P_z matrix and normalized state vector.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction3, 'context': ['user query','thinking of subtask 2','answer of subtask 2'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, computing expectation ⟨P_z⟩, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_sc_instruction4 = 'Sub-task 4: Compute the expectation value ⟨P_z^2⟩ = ψ† (P_z)^2 ψ by squaring the P_z matrix and evaluating with the state vector.'
    N4 = self.max_sc
    sc_agents4 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers4 = []
    thinkingmap4 = {}
    answermap4 = {}
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_sc_instruction4, 'context': ['user query','thinking of subtask 2','answer of subtask 2'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N4):
        thinking4_i, answer4_i = await sc_agents4[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction4, is_sub_task=True)
        agents.append(f'CoT-SC agent {sc_agents4[i].id}, computing expectation ⟨P_z^2⟩, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
        possible_answers4.append(answer4_i.content)
        thinkingmap4[answer4_i.content] = thinking4_i
        answermap4[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinkingmap4[answer4_content]
    answer4 = answermap4[answer4_content]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    cot_reflect_instruction5 = 'Sub-task 5: Calculate the uncertainty ΔP_z = sqrt(⟨P_z^2⟩ − ⟨P_z⟩^2) using the results from Sub-task 3 and Sub-task 4.'
    cot_agent5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max5 = self.max_round
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': cot_reflect_instruction5, 'context': ['user query','thinking of subtask 3','answer of subtask 3','thinking of subtask 4','answer of subtask 4'], 'agent_collaboration': 'Reflexion'}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking3, answer3, thinking4, answer4], cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent5.id}, calculating uncertainty, thinking: {thinking5.content}; answer: {answer5.content}')
    for i in range(N_max5):
        feedback_i, correct_i = await critic_agent5([taskInfo, thinking5, answer5], 'Please review the calculation of ΔP_z and provide any necessary corrections.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent5.id}, providing feedback, thinking: {feedback_i.content}; answer: {correct_i.content}')
        if correct_i.content == 'True':
            break
        thinking5, answer5 = await cot_agent5([taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, feedback_i], cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent5.id}, refining uncertainty calculation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    debate_instruction6 = 'Sub-task 6: Match the computed value of ΔP_z to one of the provided multiple-choice options: choice1: ħ, choice2: ħ/2, choice3: ħ/√2, choice4: √2ħ.'
    debate_agents6 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = [[] for _ in range(self.max_round)]
    all_answer6 = [[] for _ in range(self.max_round)]
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': debate_instruction6, 'context': ['user query','thinking of subtask 5','answer of subtask 5'], 'agent_collaboration': 'Debate'}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents6):
            if r == 0:
                thinking6_i, answer6_i = await agent([taskInfo, thinking5, answer5], debate_instruction6, r, is_sub_task=True)
            else:
                inputs6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6_i, answer6_i = await agent(inputs6, debate_instruction6, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, matching option, thinking: {thinking6_i.content}; answer: {answer6_i.content}')
            all_thinking6[r].append(thinking6_i)
            all_answer6[r].append(answer6_i)
    final_decision_agent6 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent6([taskInfo] + all_thinking6[-1] + all_answer6[-1], 'Sub-task 6: Make final decision on the correct multiple-choice option for ΔP_z.', is_sub_task=True)
    agents.append(f'Final Decision agent, selecting correct option, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs