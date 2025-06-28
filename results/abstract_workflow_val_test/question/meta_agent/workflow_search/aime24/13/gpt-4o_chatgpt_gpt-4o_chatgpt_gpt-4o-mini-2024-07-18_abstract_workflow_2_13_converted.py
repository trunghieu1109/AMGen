async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning
    
    # Sub-task 1: Determine the configuration and properties of the eight circles of radius 34.
    cot_instruction_1 = "Sub-task 1: Determine the configuration and properties of the eight circles of radius 34 that are tangent to each other and to the sides of triangle ABC."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determining configuration of circles, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine the configuration and properties of the 2024 circles of radius 1.
    cot_instruction_2 = "Sub-task 2: Determine the configuration and properties of the 2024 circles of radius 1 that are tangent to each other and to the sides of triangle ABC."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, determining configuration of smaller circles, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Establish the relationship between the configuration of the circles and the sides of triangle ABC.
    cot_sc_instruction_3 = "Sub-task 3: Establish the relationship between the configuration of the circles and the sides of triangle ABC."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}

    for i in range(N):
        thinking3, answer3 = cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, establishing relationship, thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3

    answer3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3]
    answer3 = answermapping_3[answer3]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Intermediate Inference and Validation

    # Sub-task 4: Calculate the side lengths of triangle ABC.
    cot_reflect_instruction_4 = "Sub-task 4: Calculate the side lengths of triangle ABC using the information from the circle configurations."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round

    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, calculating side lengths, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback4, correct4 = critic_agent_4([taskInfo, thinking4, answer4], "please review the side lengths calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}')
        if correct4.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining side lengths, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Determine the semi-perimeter of triangle ABC.
    cot_instruction_5 = "Sub-task 5: Determine the semi-perimeter of triangle ABC using the side lengths."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, calculating semi-perimeter, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Calculate the area of triangle ABC.
    cot_reflect_instruction_6 = "Sub-task 6: Calculate the area of triangle ABC using the circle configurations and side lengths."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round

    cot_inputs_6 = [taskInfo, thinking4, answer4]
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, calculating area, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max_6):
        feedback6, correct6 = critic_agent_6([taskInfo, thinking6, answer6], "please review the area calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}')
        if correct6.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining area, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Stage 3: Final Output Generation and Integration

    # Sub-task 7: Compute the inradius of triangle ABC.
    cot_instruction_7 = "Sub-task 7: Compute the inradius of triangle ABC using the area and semi-perimeter."
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = cot_agent_7([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_7.id}, calculating inradius, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Express the inradius as a fraction in simplest form and find the sum of the numerator and denominator.
    debate_instruction_8 = "Sub-task 8: Express the inradius as a fraction in simplest form and find the sum of the numerator and denominator."
    debate_agents_8 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round

    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]

    for r in range(N_max_8):
        for agent in debate_agents_8:
            input_infos_8 = [taskInfo, thinking7, answer7]
            if r > 0:
                input_infos_8.extend(all_thinking8[r-1])
            thinking8, answer8 = agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, expressing inradius, thinking: {thinking8.content}; answer: {answer8.content}')
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)

    final_answers_8 = [ans.content for ans in all_answer8[-1]]
    final_answercontent_8 = Counter(final_answers_8).most_common(1)[0][0]
    index_8 = final_answers_8.index(final_answercontent_8)
    thinking8 = all_thinking8[-1][index_8]
    answer8 = all_answer8[-1][index_8]
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer