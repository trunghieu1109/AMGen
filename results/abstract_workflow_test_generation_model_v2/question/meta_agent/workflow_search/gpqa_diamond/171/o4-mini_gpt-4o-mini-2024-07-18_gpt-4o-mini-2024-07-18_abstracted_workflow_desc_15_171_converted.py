async def forward_171(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = 'Sub-task 1: Identify the governing Boltzmann excitation formula under LTE that relates the population of a specific energy level to the temperature in a stellar photosphere.'
    cot_agent1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction1, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1.id}, identifying Boltzmann formula, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction2 = 'Sub-task 2: Apply the Boltzmann formula to both stars, form the ratio of the excited-level populations (star_1 over star_2), and set it equal to 2.'
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping2 = {}
    answermapping2 = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction2, 'context': ['user query','response of subtask 1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents2[i].id}, applying Boltzmann ratio, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping2[answer2.content] = thinking2
        answermapping2[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping2[answer2_content]
    answer2 = answermapping2[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_reflect_instruction3 = 'Sub-task 3: Algebraically manipulate the ratio equation from subtask_2 to isolate ln(2) and express it in terms of T_1 and T_2.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max3 = self.max_round
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_reflect_instruction3, 'context': ['user query','response of subtask 1','response of subtask 2'], 'agent_collaboration': 'Reflexion'}
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent3.id}, manipulating equation, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max3):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], 'please review the algebraic manipulation and provide limitations.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}')
        if correct3.content == 'True':
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent3.id}, refining manipulation, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    debate_instruction4 = 'Sub-task 4: Compare the derived ln(2) expression to the four provided choices and determine which one matches exactly.'
    debate_agents4 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max4)]
    all_answer4 = [[] for _ in range(N_max4)]
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': debate_instruction4, 'context': ['user query','response of subtask 3'], 'agent_collaboration': 'Debate'}
    for r in range(N_max4):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                input_infos4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos4, debate_instruction4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, comparing choices, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent4 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo] + all_thinking4[-1] + all_answer4[-1], 'Sub-task 4: Make final decision on correct choice.', is_sub_task=True)
    agents.append(f'Final Decision agent, choosing correct equation, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs