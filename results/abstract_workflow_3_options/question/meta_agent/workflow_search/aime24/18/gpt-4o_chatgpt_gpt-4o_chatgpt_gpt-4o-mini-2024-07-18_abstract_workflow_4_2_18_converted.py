async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1: Core Information and Constraint Extraction
    # Sub-task 1: Identify the coordinates of points O, A, and B, and the properties of segment AB.
    cot_instruction_1 = "Sub-task 1: Identify the coordinates of points O, A, and B, and the properties of segment AB."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determine coordinates and properties of segment AB, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify the properties and constraints of the family of segments F.
    cot_instruction_2 = "Sub-task 2: Identify the properties and constraints of the family of segments F, including the condition that P is on the x-axis and Q is on the y-axis."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, identifying properties of segments F, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify the condition for the unique point C on segment AB.
    cot_instruction_3 = "Sub-task 3: Identify the condition that there is a unique point C on segment AB that does not belong to any segment from F other than AB itself."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, determining condition for unique point C, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Parametric Representation Derivation
    # Sub-task 4: Derive the parametric representation of segment AB.
    cot_instruction_4 = "Sub-task 4: Derive the parametric representation of segment AB."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, deriving parametric representation of segment AB, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Derive the parametric representation of the segments in family F.
    cot_instruction_5 = "Sub-task 5: Derive the parametric representation of the segments in family F."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent_5([taskInfo, thinking2, answer2], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, deriving parametric representation of segments in F, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Stage 3: Expression Transformation and Alignment
    # Sub-task 6: Transform the parametric representation of segment AB to find the coordinates of point C.
    cot_reflect_instruction_6 = "Sub-task 6: Transform the parametric representation of segment AB to find the coordinates of point C."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_6 = [taskInfo, thinking3, answer3, thinking4, answer4]
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, transforming parametric representation to find C, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max):
        feedback, correct = critic_agent_6([taskInfo, thinking6, answer6], "please review the transformation to find C and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining transformation to find C, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Analyze the parametric representation of segments in F to ensure point C does not belong to any segment other than AB.
    cot_instruction_7 = "Sub-task 7: Analyze the parametric representation of segments in F to ensure point C does not belong to any segment other than AB."
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = cot_agent_7([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_7.id}, analyzing segments in F, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    # Stage 4: Final Output Calculation
    # Sub-task 8: Calculate OC^2 and express it as a fraction p/q where p and q are relatively prime.
    debate_instruction_8 = "Sub-task 8: Calculate OC^2 and express it as a fraction p/q where p and q are relatively prime."
    debate_agents_8 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.4) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]

    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            input_infos_8 = [taskInfo, thinking6, answer6, thinking7, answer7]
            if r > 0:
                input_infos_8.extend(all_thinking8[r-1])
            thinking8, answer8 = agent(input_infos_8, debate_instruction_8, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating OC^2, thinking: {thinking8.content}; answer: {answer8.content}')
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)

    final_decision_agent_8 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking8, answer8 = final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make a final decision on OC^2.", is_sub_task=True)
    agents.append(f'Final Decision agent on calculating OC^2, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print("Step 8: ", sub_tasks[-1])

    # Sub-task 9: Find the sum p + q.
    cot_instruction_9 = "Sub-task 9: Find the sum p + q."
    cot_agent_9 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking9, answer9 = cot_agent_9([taskInfo, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_9.id}, finding sum p + q, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer