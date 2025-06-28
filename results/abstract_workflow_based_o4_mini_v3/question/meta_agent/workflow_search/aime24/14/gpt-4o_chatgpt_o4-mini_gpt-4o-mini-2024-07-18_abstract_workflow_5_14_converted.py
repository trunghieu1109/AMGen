async def forward_14(self, taskInfo):
    sub_tasks = []
    agents = []

    # Stage 1: Problem Decomposition and Information Extraction
    # Sub-task 1
    cot_instruction_1 = 'Sub-task 1: Identify and understand the properties of the hyperbola x^2/20 - y^2/24 = 1, including its axes, asymptotes, and how points are generally represented on this hyperbola.'
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determine properties of hyperbola, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Subtask 1 answer: ', sub_tasks[-1])

    # Sub-task 2
    cot_instruction_2 = 'Sub-task 2: Understand the properties of a rhombus, focusing on how the diagonals intersect and how this applies when one of the intersections is at the origin.'
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, determine properties of rhombus, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Subtask 2 answer: ', sub_tasks[-1])

    # Sub-task 3
    cot_instruction_3 = 'Sub-task 3: Determine the parametric equations or coordinates for points A, B, C, and D such that they lie on the given hyperbola and form the vertices of a rhombus with diagonals intersecting at the origin.'
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, determine coordinates of rhombus vertices, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Subtask 3 answer: ', sub_tasks[-1])

    # Stage 2: Intermediate Value Analysis and Transformation
    # Sub-task 4
    cot_reflect_instruction_4 = 'Sub-task 4: Calculate the expressions for the diagonal lengths AC and BD of the rhombus using the coordinates identified in subtask 3. Recognize how these relate to the properties of the hyperbola.'
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]

    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, calculation of diagonal lengths, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], 'please review the diagonal calculation and provide its limitations.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback on diagonal length calculation, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refinement of diagonal lengths, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Subtask 4 answer: ', sub_tasks[-1])

    # Sub-task 5
    debate_instruction_5 = 'Sub-task 5: Express BD^2 based on the calculations from subtask 4 and establish any inequalities or constraints that limit its maximum possible value.'
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            input_infos_5 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos_5.extend(all_thinking5[r-1])
            thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, defining BD^2 constraints, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make final decision on BD^2 constraints.', is_sub_task=True)
    agents.append(f'Final Decision agent, defining BD^2 constraints, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Subtask 5 answer: ', sub_tasks[-1])

    # Stage 3: Final Output Computation
    # Sub-task 6
    cot_instruction_6 = 'Sub-task 6: Determine the conditions or configurations when BD^2 is maximized, such that all points still comply with the properties of the hyperbola and the rhombus.'
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, finding max configurations for BD^2, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Subtask 6 answer: ', sub_tasks[-1])

    # Stage 4: Final Answer Derivation and Formatting
    # Sub-task 7
    reflex_instruction_7 = 'Sub-task 7: Convert the final results of the previous subtask into the greatest real number that is less than the maximum calculated BD^2.'
    final_decision_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo, thinking6, answer6], reflex_instruction_7, is_sub_task=True)
    agents.append(f'Final Decision agent, calculating greatest real number of BD^2, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Subtask 7 answer: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer