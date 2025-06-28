async def forward_3(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []

    cot_instruction = 'Sub-task 1: Compile the four original Maxwell’s equations in differential form assuming no magnetic monopoles.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, compiling original equations, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])

    cot_instruction2 = 'Sub-task 2: From the compiled equations in Sub-task 1, isolate and restate explicitly Gauss’s law for magnetism (divergence of B = 0) and Faraday’s law (curl of E = -∂B/∂t).'
    cot_agent2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2.id}, restating key laws, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])

    sc_instruction3 = 'Sub-task 3: Introduce magnetic charge density rho_m and current density J_m, write the modified Maxwell’s equations in differential form showing new monopole terms.'
    cot_agents3 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    for agent in cot_agents3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking1, answer1], sc_instruction3, is_sub_task=True)
        agents.append(f'CoT-SC agent {agent.id}, introducing monopole terms, thinking: {thinking3_i.content}; answer: {answer3_i.content}')
        possible_answers.append(answer3_i.content)
        thinking_map[answer3_i.content] = thinking3_i
        answer_map[answer3_i.content] = answer3_i
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinking_map[most_common_answer]
    answer3 = answer_map[most_common_answer]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])

    cot_instruction4 = 'Sub-task 4: Compare modified equations from Sub-task 3 with original ones in Sub-task 2 and identify which equations have additional monopole terms.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, comparing equations, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])

    cot_instruction5 = 'Sub-task 5: Translate each provided answer choice into statements of which Maxwell equations they claim to differ.'
    cot_agent5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent5.id}, translating choices, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5: ', sub_tasks[-1])

    debate_instruction6 = 'Sub-task 6: Based on the actual modified equations from Sub-task 4 and the choice interpretations from Sub-task 5, determine which choice matches exactly.'
    debate_agents6 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = [[] for _ in range(self.max_round)]
    all_answer6 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents6):
            if r == 0:
                thinking6_i, answer6_i = await agent([taskInfo, thinking4, answer4, thinking5, answer5], debate_instruction6, r, is_sub_task=True)
            else:
                inputs6 = [taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6_i, answer6_i = await agent(inputs6, debate_instruction6, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, discussing choice matching, thinking: {thinking6_i.content}; answer: {answer6_i.content}')
            all_thinking6[r].append(thinking6_i)
            all_answer6[r].append(answer6_i)
    final_decision_agent6 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent6([taskInfo] + all_thinking6[-1] + all_answer6[-1], 'Sub-task 6: Make final decision on correct choice.', is_sub_task=True)
    agents.append(f'Final Decision agent, making final choice, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer