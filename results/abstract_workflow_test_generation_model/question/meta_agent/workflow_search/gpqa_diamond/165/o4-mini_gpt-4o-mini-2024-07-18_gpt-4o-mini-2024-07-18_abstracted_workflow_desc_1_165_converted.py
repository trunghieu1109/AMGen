async def forward_165(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Identify all fields in the extended SM Lagrangian, their gauge quantum numbers, and VEVs.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identify fields and VEVs, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1:', sub_tasks[-1])
    N = self.max_sc
    sc_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmap = {}
    answermap = {}
    sc_instruction = 'Sub-task 2: Determine which global U(1) is broken by x != 0 and identify the pseudo-Goldstone boson direction.'
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': sc_instruction, 'context': ['user query','thinking1','answer1'], 'agent_collaboration': 'SC_CoT'}
    for agent in sc_agents:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, find broken U(1) and Goldstone direction, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmap[answer2.content] = thinking2
        answermap[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmap[answer2_content]
    answer2 = answermap[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2:', sub_tasks[-1])
    M = self.max_sc
    pot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(M)]
    pot_answers = []
    pot_thinkingmap = {}
    pot_answermap = {}
    pot_instruction = 'Sub-task 3: Write down the one-loop Coleman-Weinberg effective potential V1(phi,h) including all relevant contributions with multiplicities and couplings.'
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': pot_instruction, 'context': ['user query','thinking2','answer2'], 'agent_collaboration': 'SC_CoT'}
    for agent in pot_agents:
        thinking3, answer3 = await agent([taskInfo, thinking2, answer2], pot_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, write one-loop effective potential, thinking: {thinking3.content}; answer: {answer3.content}")
        pot_answers.append(answer3.content)
        pot_thinkingmap[answer3.content] = thinking3
        pot_answermap[answer3.content] = answer3
    answer3_content = Counter(pot_answers).most_common(1)[0][0]
    thinking3 = pot_thinkingmap[answer3_content]
    answer3 = pot_answermap[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3:', sub_tasks[-1])
    cot_agent2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    instruction4 = 'Sub-task 4: Compute the mass squared of H2 by taking the second derivative of V1 along the pseudo-Goldstone direction at theta=0.'
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': instruction4, 'context': ['user query','thinking3','answer3'], 'agent_collaboration': 'Reflexion'}
    inputs4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent2(inputs4, instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, initial mass squared computation, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic([taskInfo, thinking4, answer4], 'Review the second derivative computation and point out any issues.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == 'True':
            break
        inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent2(inputs4, instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refinement {i+1}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4:', sub_tasks[-1])
    cot_instruction5 = 'Sub-task 5: Recast the derived M_h2^2 expression into schematic form identifying positive/negative contributions and counting alpha_i coefficients.'
    cot_agent5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': cot_instruction5, 'context': ['user query','thinking4','answer4'], 'agent_collaboration': 'CoT'}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, recast mass expression, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5:', sub_tasks[-1])
    N6 = self.max_sc
    comp_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N6)]
    comp_answers = []
    comp_thinkingmap = {}
    comp_answermap = {}
    instruction6 = 'Sub-task 6: Compare the schematic formula against choices A-D, checking presence of top-quark, A0 term, and factor placement.'
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': instruction6, 'context': ['user query','thinking5','answer5'], 'agent_collaboration': 'SC_CoT'}
    for agent in comp_agents:
        thinking6, answer6 = await agent([taskInfo, thinking5, answer5], instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, compare formula to choices, thinking: {thinking6.content}; answer: {answer6.content}")
        comp_answers.append(answer6.content)
        comp_thinkingmap[answer6.content] = thinking6
        comp_answermap[answer6.content] = answer6
    answer6_content = Counter(comp_answers).most_common(1)[0][0]
    thinking6 = comp_thinkingmap[answer6_content]
    answer6 = comp_answermap[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6:', sub_tasks[-1])
    debate_agents = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = [[] for _ in range(self.max_round)]
    all_answer7 = [[] for _ in range(self.max_round)]
    instruction7 = 'Sub-task 7: Debate and decide the final choice letter for M_h2^2 expression from A, B, C, or D.'
    subtask_desc7 = {'subtask_id': 'subtask_7', 'instruction': instruction7, 'context': ['user query','thinking6','answer6'], 'agent_collaboration': 'Debate'}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], instruction7, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(inputs7, instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final choice, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision([taskInfo] + all_thinking7[-1] + all_answer7[-1], 'Sub-task 7: Make final decision on choice letter.', is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {'thinking': thinking7, 'answer': answer7}
    logs.append(subtask_desc7)
    print('Step 7:', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs