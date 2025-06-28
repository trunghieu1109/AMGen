async def forward_21(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1: Problem Decomposition and Information Extraction
    
    # Sub-task 1: Identify vertices of the regular dodecagon
    cot_instruction_1 = 'Sub-task 1: Identify vertices of the regular dodecagon and represent them in a coordinate system.'
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, vertices identification, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Subtask 1 answer: ', sub_tasks[-1])

    # Sub-task 2: Determine diagonal types and lengths
    cot_instruction_2 = 'Sub-task 2: Determine different diagonal types and their lengths in the regular dodecagon.'
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, diagonal types and lengths, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Subtask 2 answer: ', sub_tasks[-1])

    # Sub-task 3: Identify conditions for rectangle formation
    cot_reflect_instruction_3 = 'Sub-task 3: Determine conditions under which four vertices form a rectangle.'
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, conditions for rectangles, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Subtask 3 answer: ', sub_tasks[-1])

    # Stage 2: Intermediate Value Analysis
    
    # Sub-task 4: Analyze and pair vertices forming rectangles
    debate_instruction_4 = 'Sub-task 4: Analyze and pair vertices that form potential rectangles in the dodecagon.'
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking2, answer2, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, analyzing vertex pairings, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], 'Sub-task 4: Final decision on vertex pairings.', is_sub_task=True)
    agents.append(f'Final Decision agent, vertex pairings, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Subtask 4 answer: ', sub_tasks[-1])

    # Sub-task 5: Calculate number of distinct rectangles
    cot_instruction_5 = 'Sub-task 5: Calculate the number of distinct rectangles based on the vertex pairings.'
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, calculating number of rectangles, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Subtask 5 answer: ', sub_tasks[-1])

    # Stage 3: Validation
    
    # Sub-task 6: Validate rectangle count
    cot_reflect_instruction_6 = 'Sub-task 6: Validate the calculated number of rectangles by ensuring all conditions are met.'
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, validating rectangle count, thinking: {thinking6.content}; answer: {answer6.content}')
    for i in range(N_max_6):
        feedback, correct = await critic_agent_3([taskInfo, thinking6, answer6], 'please confirm if the rectangle count is valid and correct.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, feedback on rectangle count, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining rectangle count validation, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Subtask 6 answer: ', sub_tasks[-1])

    # Stage 4: Final Answer Derivation and Formatting
    
    # Sub-task 7: Format the final count of rectangles
    cot_instruction_7 = 'Sub-task 7: Format the final count into the required response format and verify requirements.'
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_7.id}, formatting final count, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Subtask 7 answer: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
