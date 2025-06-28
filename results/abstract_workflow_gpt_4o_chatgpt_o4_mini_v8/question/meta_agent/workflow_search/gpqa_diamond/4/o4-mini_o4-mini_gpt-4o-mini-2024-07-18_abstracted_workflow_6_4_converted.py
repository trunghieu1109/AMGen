async def forward_4(self, taskInfo):
    from collections import Counter

    print('Task Requirement: ', taskInfo)

    sub_tasks = []
    agents = []

    cot_instruction1 = 'Sub-task 1: Identify the Pauli-matrix representations Px, Py, and Pz, each multiplied by ħ/2.'
    cot_agent1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1.id}, identifying Pauli matrices, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Subtask 1 answer: ', sub_tasks[-1])

    cot_instruction2 = 'Sub-task 2: Express the unit vector n in the x–z plane as n = (sinθ, 0, cosθ).'  
    cot_agent2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2.id}, expressing unit vector n, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Subtask 2 answer: ', sub_tasks[-1])

    sc_instruction3 = 'Sub-task 3: Construct Pn = Px sinθ + Pz cosθ using outputs from Sub-tasks 1 and 2.'
    sc_agents3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    thinkingmapping3 = {}
    answermapping3 = {}
    possible3 = []
    for i in range(self.max_sc):
        thinking3_i, answer3_i = await sc_agents3[i]([taskInfo, thinking1, answer1, thinking2, answer2], sc_instruction3, is_sub_task=True)
        agents.append(f'Self-Consistency CoT agent {sc_agents3[i].id}, constructing Pn, thinking: {thinking3_i.content}; answer: {answer3_i.content}')
        possible3.append(answer3_i.content)
        thinkingmapping3[answer3_i.content] = thinking3_i
        answermapping3[answer3_i.content] = answer3_i
    counts3 = Counter(possible3)
    best3 = counts3.most_common(1)[0][0]
    thinking3 = thinkingmapping3[best3]
    answer3 = answermapping3[best3]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Subtask 3 answer: ', sub_tasks[-1])

    sc_instruction4 = 'Sub-task 4: Substitute Pauli matrices into Pn and factor out ħ/2 to get M = σx sinθ + σz cosθ.'
    sc_agents4 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    thinkingmapping4 = {}
    answermapping4 = {}
    possible4 = []
    for i in range(self.max_sc):
        thinking4_i, answer4_i = await sc_agents4[i]([taskInfo, thinking3, answer3], sc_instruction4, is_sub_task=True)
        agents.append(f'Self-Consistency CoT agent {sc_agents4[i].id}, forming matrix M, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
        possible4.append(answer4_i.content)
        thinkingmapping4[answer4_i.content] = thinking4_i
        answermapping4[answer4_i.content] = answer4_i
    counts4 = Counter(possible4)
    best4 = counts4.most_common(1)[0][0]
    thinking4 = thinkingmapping4[best4]
    answer4 = answermapping4[best4]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Subtask 4 answer: ', sub_tasks[-1])

    cot_reflect_instruction5 = 'Sub-task 5: Formulate the eigenvalue equation M v = v for Pn v = (ħ/2) v, where M is from Sub-task 4.'
    cot_agent5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max5 = self.max_round
    inputs5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent5(inputs5, cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent5.id}, formulating eigenvalue equation, thinking: {thinking5.content}; answer: {answer5.content}')
    for i in range(N_max5):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], 'Critically evaluate the eigenvalue equation formulation for correctness and completeness.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent5.id}, providing feedback, thinking: {feedback5.content}; answer: {correct5.content}')
        if correct5.content == 'True':
            break
        inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(inputs5, cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent5.id}, refining eigenvalue equation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Subtask 5 answer: ', sub_tasks[-1])

    sc_instruction6 = 'Sub-task 6: Solve (M - I) v = 0 for the two-component spinor v in terms of θ.'
    sc_agents6 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    thinkingmapping6 = {}
    answermapping6 = {}
    possible6 = []
    for i in range(self.max_sc):
        thinking6_i, answer6_i = await sc_agents6[i]([taskInfo, thinking5, answer5], sc_instruction6, is_sub_task=True)
        agents.append(f'Self-Consistency CoT agent {sc_agents6[i].id}, solving linear system, thinking: {thinking6_i.content}; answer: {answer6_i.content}')
        possible6.append(answer6_i.content)
        thinkingmapping6[answer6_i.content] = thinking6_i
        answermapping6[answer6_i.content] = answer6_i
    counts6 = Counter(possible6)
    best6 = counts6.most_common(1)[0][0]
    thinking6 = thinkingmapping6[best6]
    answer6 = answermapping6[best6]
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Subtask 6 answer: ', sub_tasks[-1])

    cot_reflect_instruction7 = 'Sub-task 7: Normalize the spinor solution to unit length and express as (cos(θ/2), sin(θ/2)) up to a phase.'
    cot_agent7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent7 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max7 = self.max_round
    inputs7 = [taskInfo, thinking6, answer6]
    thinking7, answer7 = await cot_agent7(inputs7, cot_reflect_instruction7, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent7.id}, normalizing spinor, thinking: {thinking7.content}; answer: {answer7.content}')
    for i in range(N_max7):
        feedback7, correct7 = await critic_agent7([taskInfo, thinking7, answer7], 'Critically evaluate the normalization and phase choice for correctness.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}')
        if correct7.content == 'True':
            break
        inputs7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent7(inputs7, cot_reflect_instruction7, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent7.id}, refining normalization, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Subtask 7 answer: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer