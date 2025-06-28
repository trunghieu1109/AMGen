async def forward_4(self, taskInfo):
    from collections import Counter
    print('Task Requirement:', taskInfo)
    sub_tasks = []
    agents = []
    cot_instruction = 'Sub-task 1: Extract and record the definitions of the spin operators Px, Py, and Pz in terms of the given 2×2 Pauli matrices times ħ/2.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, extracting operator definitions, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1:', sub_tasks[-1])
    cot_instruction2 = 'Sub-task 2: Define the arbitrary unit direction vector n in the x–z plane as n = (sinθ, 0, cosθ).'
    cot_agent2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2.id}, defining direction vector, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2:', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 3: Construct the operator Pn = n_x Px + n_z Pz as a 2×2 matrix using outputs from subtasks 1 and 2.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking3_i, answer3_i = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, constructing Pn matrix, thinking: {thinking3_i.content}; answer: {answer3_i.content}')
        possible_answers.append(answer3_i.content)
        thinkingmapping[answer3_i.content] = thinking3_i
        answermapping[answer3_i.content] = answer3_i
    answer3_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[answer3_content]
    answer3 = answermapping[answer3_content]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3:', sub_tasks[-1])
    cot_reflect_instruction = 'Sub-task 4: Formulate the eigenvalue equation Pn ψ = (ħ/2) ψ in component form, writing the two linear equations for ψ components.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent4.id}, formulating eigenvalue equations, thinking: {thinking4.content}; answer: {answer4.content}')
    for i in range(N_max):
        feedback, correct = await critic_agent4([taskInfo, thinking4, answer4], 'Review the formulation of the eigenvalue equations and identify any errors.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent4.id}, refining eigenvalue equations, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4:', sub_tasks[-1])
    debate_instruction5 = 'Sub-task 5: Solve the system of two linear equations from subtask 4 to find the unnormalized eigenvector components for eigenvalue +ħ/2.'
    debate_agents5 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents5):
            input_infos5 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos5.extend(all_thinking5[r-1] + all_answer5[r-1])
            thinking5_i, answer5_i = await agent(input_infos5, debate_instruction5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, solving linear equations, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision5 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make final decision on the unnormalized eigenvector components.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision5.id}, selecting unnormalized components, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5:', sub_tasks[-1])
    cot_instruction6 = 'Sub-task 6: Normalize the eigenvector from subtask 5 so that its norm is unity.'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, normalizing eigenvector, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6:', sub_tasks[-1])
    final_decision7 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision7([taskInfo, thinking6, answer6], 'Sub-task 7: Express the final normalized eigenvector in terms of θ and match against the provided answer choices.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision7.id}, matching against choices, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Step 7:', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer