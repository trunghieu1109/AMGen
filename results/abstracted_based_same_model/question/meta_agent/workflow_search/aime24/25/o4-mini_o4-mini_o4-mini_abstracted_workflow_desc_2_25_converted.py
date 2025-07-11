async def forward_25(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = 'Sub-task 1: Define the three exterior turning angles α, β, γ at consecutive vertices A→B, B→C, and C→D of the equilateral parallelohexagon and derive the relation α + β + γ = π from the fact that the six exterior angles sum to 2π and they repeat twice around the hexagon.'
    cot_agent1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction1, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1.id}, defining angles and relation α+β+γ=π, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1:', sub_tasks[-1])
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    cot_instruction2 = 'Sub-task 2: Choose a coordinate system (e.g., A=(0,0), AB along the x-axis) and express the six edge vectors AB, BC, CD, DE, EF, FA in terms of the common side length s and the exterior angles α, β, γ.'
    cot_agent2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_instruction2, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'], 'agent_collaboration': 'CoT'}
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2.id}, expressing edge vectors, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2:', sub_tasks[-1])
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    cot_instruction3 = 'Sub-task 3: From the vector expressions of the edges, write the explicit line equations for lines AB, CD, and EF in the chosen coordinate system.'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction3, 'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, writing line equations, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3:', sub_tasks[-1])
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    cot_instruction4 = 'Sub-task 4: Compute symbolically the intersection points P = AB∩CD, Q = CD∩EF, and R = EF∩AB using the line equations from Subtask 3.'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_instruction4, 'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3'], 'agent_collaboration': 'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, computing intersection points, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4:', sub_tasks[-1])
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    cot_sc_instruction5 = 'Sub-task 5: Derive formulas for the distances PQ, QR, and RP of triangle PQR in terms of s, α, β, γ based on the coordinates of P, Q, R.'
    N = self.max_sc
    cot_agents5 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers5 = []
    thinkingmapping5 = {}
    answermapping5 = {}
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': cot_sc_instruction5, 'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking5, answer5 = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents5[i].id}, deriving distance formulas, thinking: {thinking5.content}; answer: {answer5.content}')
        possible_answers5.append(answer5.content)
        thinkingmapping5[answer5.content] = thinking5
        answermapping5[answer5.content] = answer5
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinkingmapping5[answer5_content]
    answer5 = answermapping5[answer5_content]
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5:', sub_tasks[-1])
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    cot_instruction6a = 'Sub-task 6a: Express the angles of triangle PQR (∠P, ∠Q, ∠R) explicitly in terms of α, β, γ (e.g., ∠P = π−β, ∠Q = π−γ, ∠R = π−α).'
    cot_agent6a = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6a = {'subtask_id': 'subtask_6a', 'instruction': cot_instruction6a, 'context': ['user query', 'thinking of subtask 5', 'answer of subtask 5'], 'agent_collaboration': 'CoT'}
    thinking6a, answer6a = await cot_agent6a([taskInfo, thinking5, answer5], cot_instruction6a, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6a.id}, expressing triangle angles, thinking: {thinking6a.content}; answer: {answer6a.content}')
    sub_tasks.append(f'Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}')
    print('Step 6:', sub_tasks[-1])
    subtask_desc6a['response'] = {'thinking': thinking6a, 'answer': answer6a}
    logs.append(subtask_desc6a)
    cot_sc_instruction6b = 'Sub-task 6b: Propose possible mappings of triangle angles ∠P, ∠Q, ∠R to expressions in α, β, γ, and verify which mapping satisfies α+β+γ=π and yields consistent side ratios PQ:QR:RP.'
    N = self.max_sc
    cot_agents6b = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers6b = []
    thinkingmapping6b = {}
    answermapping6b = {}
    subtask_desc6b = {'subtask_id': 'subtask_6b', 'instruction': cot_sc_instruction6b, 'context': ['user query', 'thinking of subtask 6a', 'answer of subtask 6a'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking6b, answer6b = await cot_agents6b[i]([taskInfo, thinking6a, answer6a], cot_sc_instruction6b, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents6b[i].id}, proposing and verifying mappings, thinking: {thinking6b.content}; answer: {answer6b.content}')
        possible_answers6b.append(answer6b.content)
        thinkingmapping6b[answer6b.content] = thinking6b
        answermapping6b[answer6b.content] = answer6b
    answer6b_content = Counter(possible_answers6b).most_common(1)[0][0]
    thinking6b = thinkingmapping6b[answer6b_content]
    answer6b = answermapping6b[answer6b_content]
    sub_tasks.append(f'Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}')
    print('Step 7:', sub_tasks[-1])
    subtask_desc6b['response'] = {'thinking': thinking6b, 'answer': answer6b}
    logs.append(subtask_desc6b)
    cot_reflect_instruction6c = 'Sub-task 6c: Apply the Law of Sines to triangle PQR with PQ=200, QR=240, RP=300 and the verified angle mapping to derive equations for α, β, γ under the constraint α+β+γ=π and solve for these angles.'
    cot_agent6c = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent6c = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs6c = [taskInfo, thinking6b, answer6b]
    subtask_desc6c = {'subtask_id': 'subtask_6c', 'instruction': cot_reflect_instruction6c, 'context': ['user query', 'thinking of subtask 6b', 'answer of subtask 6b'], 'agent_collaboration': 'Reflexion'}
    thinking6c, answer6c = await cot_agent6c(cot_inputs6c, cot_reflect_instruction6c, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent6c.id}, initial angle solving, thinking: {thinking6c.content}; answer: {answer6c.content}')
    for i in range(N_max):
        feedback6c, correct6c = await critic_agent6c([taskInfo, thinking6c, answer6c], 'please review the derived equations for α, β, γ and provide feedback.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent6c.id}, feedback: {feedback6c.content}; correct: {correct6c.content}')
        if correct6c.content == 'True':
            break
        cot_inputs6c.extend([thinking6c, answer6c, feedback6c])
        thinking6c, answer6c = await cot_agent6c(cot_inputs6c, cot_reflect_instruction6c, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent6c.id}, refining angle solving, thinking: {thinking6c.content}; answer: {answer6c.content}')
    sub_tasks.append(f'Sub-task 6c output: thinking - {thinking6c.content}; answer - {answer6c.content}')
    print('Step 8:', sub_tasks[-1])
    subtask_desc6c['response'] = {'thinking': thinking6c, 'answer': answer6c}
    logs.append(subtask_desc6c)
    debate_instruction7 = 'Sub-task 7: Substitute the solved values of α, β, γ into one of the distance formulas from Subtask 5 to compute the numerical value of the hexagon side length s.'
    debate_agents7 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max7)]
    all_answer7 = [[] for _ in range(N_max7)]
    subtask_desc7 = {'subtask_id': 'subtask_7', 'instruction': debate_instruction7, 'context': ['user query', 'thinking of subtask 6c', 'answer of subtask 6c'], 'agent_collaboration': 'Debate'}
    for r in range(N_max7):
        for i, agent in enumerate(debate_agents7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6c, answer6c], debate_instruction7, r, is_sub_task=True)
            else:
                input_infos7 = [taskInfo, thinking6c, answer6c] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos7, debate_instruction7, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, computing s, thinking: {thinking7.content}; answer: {answer7.content}')
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent7 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], 'Sub-task 7: Make final decision on the hexagon side length s.', is_sub_task=True)
    agents.append(f'Final Decision agent, determining s, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Step 9:', sub_tasks[-1])
    subtask_desc7['response'] = {'thinking': thinking7, 'answer': answer7}
    logs.append(subtask_desc7)
    cot_reflect_instruction8 = 'Sub-task 8: Verify that the computed side length s and angles α, β, γ satisfy the geometric conditions: parallel opposite sides, equilateral side length, PQ=200, QR=240, RP=300, and closure of the hexagon.'
    cot_agent8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent8 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max8 = self.max_round
    cot_inputs8 = [taskInfo, thinking7, answer7]
    subtask_desc8 = {'subtask_id': 'subtask_8', 'instruction': cot_reflect_instruction8, 'context': ['user query', 'thinking of subtask 7', 'answer of subtask 7'], 'agent_collaboration': 'Reflexion'}
    thinking8, answer8 = await cot_agent8(cot_inputs8, cot_reflect_instruction8, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent8.id}, initial validation, thinking: {thinking8.content}; answer: {answer8.content}')
    for i in range(N_max8):
        feedback8, correct8 = await critic_agent8([taskInfo, thinking8, answer8], 'please review the verification of computed s and angles, noting any inconsistencies.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent8.id}, feedback: {feedback8.content}; correct: {correct8.content}')
        if correct8.content == 'True':
            break
        cot_inputs8.extend([thinking8, answer8, feedback8])
        thinking8, answer8 = await cot_agent8(cot_inputs8, cot_reflect_instruction8, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent8.id}, refined validation, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print('Step 10:', sub_tasks[-1])
    subtask_desc8['response'] = {'thinking': thinking8, 'answer': answer8}
    logs.append(subtask_desc8)
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs