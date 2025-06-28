async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1: Core Information Extraction
    # Sub-task 1: Identify the relationship between the eight circles of radius 34 and the triangle ABC.
    cot_instruction_1 = "Sub-task 1: Identify the relationship between the eight circles of radius 34 and the triangle ABC, focusing on how they are tangent to sides AB and BC."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determine relationship of circles with triangle ABC, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify the arrangement of 2024 circles of radius 1.
    cot_instruction_2 = "Sub-task 2: Based on the output of sub-task 1, identify the arrangement of 2024 circles of radius 1 and how they relate to the triangle ABC."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, determine arrangement of smaller circles, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Extract necessary geometric properties and constraints of triangle ABC.
    cot_instruction_3 = "Sub-task 3: Extract the necessary geometric properties and constraints of triangle ABC, such as side lengths, angles, and the relationship with the circles."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, extract geometric properties of triangle ABC, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Intermediate Value Computation
    # Sub-task 4: Calculate the side lengths of triangle ABC.
    cot_reflect_instruction_4 = "Sub-task 4: Calculate the side lengths of triangle ABC using the information about the circles and their tangency points."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, calculating side lengths of triangle ABC, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], "please review the side lengths calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining side lengths of triangle ABC, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compute the semi-perimeter of triangle ABC.
    cot_instruction_5 = "Sub-task 5: Compute the semi-perimeter of triangle ABC using the side lengths obtained."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, compute semi-perimeter of triangle ABC, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Determine the inradius of triangle ABC.
    cot_instruction_6 = "Sub-task 6: Determine the inradius of triangle ABC using the semi-perimeter and area of the triangle."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, determine inradius of triangle ABC, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Stage 3: Final Output Derivation
    # Sub-task 7: Express the inradius as a fraction m/n and calculate m+n.
    debate_instruction_7 = "Sub-task 7: Express the inradius as a fraction m/n where m and n are relatively prime, and calculate m+n."
    debate_agents_7 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.4) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            input_infos_7 = [taskInfo, thinking6, answer6]
            if r > 0:
                input_infos_7.extend(all_thinking7[r-1])
            thinking7, answer7 = agent(input_infos_7, debate_instruction_7, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, expressing inradius as fraction, thinking: {thinking7.content}; answer: {answer7.content}')
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on m+n.", is_sub_task=True)
    agents.append(f'Final Decision agent, calculating m+n, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer