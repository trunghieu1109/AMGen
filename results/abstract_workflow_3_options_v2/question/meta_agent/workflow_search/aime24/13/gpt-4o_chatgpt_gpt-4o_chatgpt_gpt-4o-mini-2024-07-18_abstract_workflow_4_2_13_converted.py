async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1: Core Information and Constraint Extraction
    # Sub-task 1: Identify the arrangement and properties of the eight circles of radius 34.
    cot_instruction_1 = "Sub-task 1: Identify the arrangement and properties of the eight circles of radius 34 and their tangency to the sides of triangle ABC."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determine properties of circles of radius 34, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify the arrangement and properties of the 2024 circles of radius 1.
    cot_instruction_2 = "Sub-task 2: Identify the arrangement and properties of the 2024 circles of radius 1 and their tangency to the sides of triangle ABC."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, determine properties of circles of radius 1, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Extract the relationship between the arrangement of circles and the inradius of triangle ABC.
    cot_sc_instruction_3 = "Sub-task 3: Extract the relationship between the arrangement of circles and the inradius of triangle ABC."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, extracting relationship, thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Parametric Representation Derivation
    # Sub-task 4: Derive the parametric representation of the inradius of triangle ABC.
    cot_instruction_4 = "Sub-task 4: Derive the parametric representation of the inradius of triangle ABC based on the arrangement of the circles."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, deriving inradius parametric representation, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Derive the parametric representation of the side lengths or other relevant parameters of triangle ABC.
    cot_instruction_5 = "Sub-task 5: Derive the parametric representation of the side lengths or other relevant parameters of triangle ABC."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent_5([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, deriving side lengths parametric representation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Stage 3: Expression Transformation and Alignment
    # Sub-task 6: Transform the parametric expressions to align with the conditions of the problem.
    cot_reflect_instruction_6 = "Sub-task 6: Transform the parametric expressions to align with the conditions of the problem, focusing on the relationship between the circle arrangements and the inradius."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_6 = [taskInfo, thinking4, answer4, thinking5, answer5]
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, transforming expressions, thinking: {thinking6.content}; answer: {answer6.content}')
    for i in range(N_max):
        feedback, correct = critic_agent_6([taskInfo, thinking6, answer6], "please review the expression transformation and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining expressions, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Stage 4: Final Output Calculation
    # Sub-task 7: Calculate the final value of the inradius of triangle ABC.
    debate_instruction_7 = "Sub-task 7: Calculate the final value of the inradius of triangle ABC and express it in the form m/n, where m and n are relatively prime."
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
            agents.append(f'Debate agent {agent.id}, round {r}, calculating inradius, thinking: {thinking7.content}; answer: {answer7.content}')
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make a final decision on the inradius.", is_sub_task=True)
    agents.append(f'Final Decision agent on calculating inradius, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Compute the sum m+n based on the calculated inradius.
    cot_instruction_8 = "Sub-task 8: Compute the sum m+n based on the calculated inradius."
    cot_agent_8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking8, answer8 = cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_8.id}, computing m+n, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer