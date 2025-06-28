async def forward_11(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning

    # Sub-task 1: Calculate the number of total paths of length 16
    cot_instruction_1 = 'Sub-task 1: Calculate the number of total paths of length 16 from the lower left to the upper right of the grid.'
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculate paths, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Sub-task 1 answer: ', sub_tasks[-1])

    # Sub-task 2: Determine what constitutes a change in direction
    cot_sc_instruction_2 = 'Sub-task 2: Based on paths calculated, determine what constitutes a change in direction.'
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, determine changes in direction, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    
    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Sub-task 2 answer: ', sub_tasks[-1])

    # Sub-task 3: Identify number of combinations with exactly four changes in direction
    cot_reflect_instruction_3 = 'Sub-task 3: Identify combinations of moves with exactly four changes in direction.'
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, identifying valid move combinations, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        feedback_3, correct_3 = await critic_agent_3([taskInfo, thinking3, answer3], 'please review and refine the move combination identification.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, reviewing combinations, thinking: {feedback_3.content}; answer: {correct_3.content}')
        if correct_3.content == 'True':
            break
        cot_inputs_3.extend([thinking3, answer3, feedback_3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining combinations, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Sub-task 3 answer: ', sub_tasks[-1])

    # Stage 2: Intermediate Inference and Validation

    # Sub-task 4: Combine data about changes in direction
    debate_instruction_4 = 'Sub-task 4: Based on valid move combinations, compute an intermediate count of paths with four direction changes.'
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round

    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for agent in debate_agents_4:
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, counting valid paths, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], 'Sub-task 4: Make final decision on path count validation.', is_sub_task=True)
    agents.append(f'Final Decision agent, validating path count, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Sub-task 4 answer: ', sub_tasks[-1])

    # Sub-task 5: Validate that intermediate path count results in exactly four changes
    cot_reflect_instruction_5 = 'Sub-task 5: Validate intermediate path count for exactly four changes.'
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round

    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, validating change count, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback_5, correct_5 = await critic_agent_5([taskInfo, thinking5, answer5], 'please review path count validation and ensure criteria met.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, reviewing changes, thinking: {feedback_5.content}; answer: {correct_5.content}')
        if correct_5.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback_5])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining validation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Sub-task 5 answer: ', sub_tasks[-1])

    # Stage 3: Final Output Generation and Integration

    # Sub-task 6: Calculate final number of valid paths
    debate_instruction_6 = 'Sub-task 6: Synthesize and finalize the number of valid paths with four direction changes.'
    debate_agents_6 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round

    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for agent in debate_agents_6:
            input_infos_6 = [taskInfo, thinking5, answer5]
            if r > 0:
                input_infos_6.extend(all_thinking6[r-1])
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, synthesizing final path count, thinking: {thinking6.content}; answer: {answer6.content}')
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], 'Sub-task 6: Finalize the number of valid paths with four direction changes.', is_sub_task=True)
    agents.append(f'Final Decision agent, finalizing path count, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Sub-task 6 answer: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
