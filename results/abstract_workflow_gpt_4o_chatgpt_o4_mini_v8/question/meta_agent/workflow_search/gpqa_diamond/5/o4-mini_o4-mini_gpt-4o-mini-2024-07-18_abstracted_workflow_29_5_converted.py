async def forward_5(self, taskInfo):
    from collections import Counter

    print('Task Requirement: ', taskInfo)

    sub_tasks = []
    agents = []

    cot_instruction = 'Sub-task 1: Convert V(r,θ)=1/2 k r^2 + 3/2 k r^2 cos^2θ into Cartesian coordinates V(x,y) using x=r cosθ and y=r sinθ.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, converting potential to Cartesian, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2: Simplify the Cartesian form V(x,y) by expanding cos^2θ = x^2/r^2 and combining like terms to express V(x,y) in the form A x^2 + B y^2.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, simplifying Cartesian form, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    selected2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[selected2]
    answer2 = answermapping[selected2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])

    cot_reflect_instruction = 'Sub-task 3: Identify the effective spring constants k_x and k_y by matching V(x,y)=1/2 k_x x^2 + 1/2 k_y y^2 to the coefficients found.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent3.id}, identifying k_x and k_y, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], 'Critically evaluate the identification of k_x and k_y for correctness and completeness.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}')
        if correct3.content == 'True':
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent3.id}, refining k_x and k_y identification, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])

    cot_instruction4 = 'Sub-task 4: Write the full Hamiltonian H=(p_x^2+p_y^2)/(2m)+1/2 k_x x^2+1/2 k_y y^2, showing it splits into two independent 1D harmonic oscillators.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, writing full Hamiltonian, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])

    debate_instruction5 = 'Sub-task 5: Write the general energy eigenvalue formula for two uncoupled oscillators: E = (n_x+1/2)ħ ω_x + (n_y+1/2)ħ ω_y.'
    debate_agents5 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for agent in debate_agents5:
            input_infos5 = [taskInfo, thinking4, answer4]
            if r>0:
                input_infos5 += all_thinking5[r-1] + all_answer5[r-1]
            thinking5, answer5 = await agent(input_infos5, debate_instruction5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent5 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo]+all_thinking5[-1]+all_answer5[-1], 'Sub-task 5: Make final decision on the energy formula.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5: ', sub_tasks[-1])

    cot_instruction6 = 'Sub-task 6: Express ω_x = sqrt(k_x/m) and ω_y = sqrt(k_y/m) in terms of the original k and m.'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking3, answer3], cot_instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, expressing angular frequencies, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6: ', sub_tasks[-1])

    cot_reflect_instruction7 = 'Sub-task 7: Substitute ω_x and ω_y into the energy formula to obtain E as a function of n_x, n_y, ħ, k, and m.'
    cot_agent7 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent7 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs7 = [taskInfo, thinking5, answer5, thinking6, answer6]
    thinking7, answer7 = await cot_agent7(cot_inputs7, cot_reflect_instruction7, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent7.id}, substituting frequencies, thinking: {thinking7.content}; answer: {answer7.content}')
    for i in range(self.max_round):
        feedback7, correct7 = await critic_agent7([taskInfo, thinking7, answer7], 'Critically evaluate the substituted energy expression.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent7.id}, feedback: {feedback7.content}; correct: {correct7.content}')
        if correct7.content=='True':
            break
        cot_inputs7 += [thinking7, answer7, feedback7]
        thinking7, answer7 = await cot_agent7(cot_inputs7, cot_reflect_instruction7, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent7.id}, refining energy substitution, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Step 7: ', sub_tasks[-1])

    cot_instruction8 = 'Sub-task 8: Compare the derived expression for E(n_x,n_y) from Sub-task 7 with each of the four multiple-choice options to find the exact match.'
    cot_agent8 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent8([taskInfo, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent8.id}, comparing with options, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print('Step 8: ', sub_tasks[-1])

    final_decision_agent9 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent9([taskInfo, thinking8, answer8], 'Sub-task 9: Select the correct answer choice for the energy spectrum.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent9.id}, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    print('Step 9: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer