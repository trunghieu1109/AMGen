async def forward_28(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Define the fixed parameters a, R, S, and set up coordinate positions for the torus circle center and sphere center to establish distance relations.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        'subtask_id': 'subtask_1',
        'instruction': cot_instruction,
        'context': ['user query'],
        'agent_collaboration': 'CoT'
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, defined fixed parameters and coordinate setup, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {
        'thinking': thinking1,
        'answer': answer1
    }
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 2: Derive the general condition for external tangency between the torus and the sphere, expressing D in terms of R, S, and tangency-circle radius r using right-triangle geometry.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {
        'subtask_id': 'subtask_2',
        'instruction': cot_sc_instruction,
        'context': ['user query', 'thinking of subtask_1', 'answer of subtask_1'],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, deriving tangency condition, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {
        'thinking': thinking2,
        'answer': answer2
    }
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_instruction_3 = 'Sub-task 3: Compute inner tangency circle radius r_i by applying the derived formula and substituting D = S - a = 11 - 3 = 8.'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        'subtask_id': 'subtask_3',
        'instruction': cot_instruction_3,
        'context': ['user query', 'thinking of subtask_2', 'answer of subtask_2'],
        'agent_collaboration': 'CoT'
    }
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, computed r_i, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {
        'thinking': thinking3,
        'answer': answer3
    }
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_instruction_4 = 'Sub-task 4: Compute outer tangency circle radius r_o by applying the derived formula and substituting D = S + a = 11 + 3 = 14.'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        'subtask_id': 'subtask_4',
        'instruction': cot_instruction_4,
        'context': ['user query', 'thinking of subtask_2', 'answer of subtask_2'],
        'agent_collaboration': 'CoT'
    }
    thinking4, answer4 = await cot_agent4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, computed r_o, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {
        'thinking': thinking4,
        'answer': answer4
    }
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    debate_instruction_5 = 'Sub-task 5: Compute Δr = r_i - r_o, simplify to m/n in lowest terms, and compute m+n.'
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        'subtask_id': 'subtask_5',
        'instruction': debate_instruction_5,
        'context': ['user query', 'thinking of subtask_3', 'answer of subtask_3', 'thinking of subtask_4', 'answer of subtask_4'],
        'agent_collaboration': 'Debate'
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, computing Δr and m+n, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make final decision on Δr and m+n.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent_5.id}, selected final m+n, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {
        'thinking': thinking5,
        'answer': answer5
    }
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs