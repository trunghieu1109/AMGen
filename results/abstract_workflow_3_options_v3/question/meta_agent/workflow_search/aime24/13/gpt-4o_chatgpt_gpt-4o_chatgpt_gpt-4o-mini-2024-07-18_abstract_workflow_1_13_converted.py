async def forward(self, taskInfo):
    from collections import Counter
    
    sub_tasks = []
    agents = []

    # Sub-task 1: Determine the configuration and arrangement of the eight circles of radius 34.
    cot_instruction_1 = "Sub-task 1: Determine the configuration and arrangement of the eight circles of radius 34 and how they are tangent to the sides of triangle ABC."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determining configuration of circles, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze the arrangement of 2024 circles of radius 1.
    cot_instruction_2 = "Sub-task 2: Analyze the arrangement of 2024 circles of radius 1 and how they can be arranged similarly to the eight circles of radius 34."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, analyzing arrangement of smaller circles, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Establish the relationship between the arrangement of the circles and the sides of triangle ABC.
    cot_sc_instruction_3 = "Sub-task 3: Establish the relationship between the arrangement of the circles and the sides of triangle ABC, focusing on the tangency conditions."
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

    # Sub-task 4: Calculate the side lengths of triangle ABC.
    cot_instruction_4 = "Sub-task 4: Calculate the side lengths of triangle ABC based on the arrangement and tangency of the circles."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, calculating side lengths, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Use the side lengths to calculate the inradius of triangle ABC.
    cot_reflect_instruction_5 = "Sub-task 5: Use the side lengths of triangle ABC to calculate its inradius."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, calculating inradius, thinking: {thinking5.content}; answer: {answer5.content}')
    for i in range(N_max_5):
        feedback5, correct5 = critic_agent_5([taskInfo, thinking5, answer5], "please review inradius calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback5.content}; answer: {correct5.content}')
        if correct5.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining inradius, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Express the inradius as a fraction and find m+n.
    cot_instruction_6 = "Sub-task 6: Express the inradius as a fraction in simplest form and find the sum of the numerator and denominator."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, expressing inradius as fraction, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer