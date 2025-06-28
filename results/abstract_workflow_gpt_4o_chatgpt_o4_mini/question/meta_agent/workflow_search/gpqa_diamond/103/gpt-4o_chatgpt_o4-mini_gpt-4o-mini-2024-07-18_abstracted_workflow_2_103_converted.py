async def forward_103(self, taskInfo):
    from collections import Counter
    
    print('Task Requirement: ', taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction_1 = 'Sub-task 1: Calculate the velocity of the star hosting planet #1 using the given periodic wavelength shift of 5 miliangstrom.'
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculating velocity for planet #1, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    
    print('Subtask 1 answer: ', sub_tasks[-1])

    cot_instruction_2 = 'Sub-task 2: Calculate the velocity of the star hosting planet #2 using the given periodic wavelength shift of 7 miliangstrom.'
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, calculating velocity for planet #2, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    print('Subtask 2 answer: ', sub_tasks[-1])

    cot_reflect_instruction_3 = 'Sub-task 3: Determine the relationship between the velocity of the star and the orbital period of the planet for planet #1 using the RV method.'
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    
    cot_inputs_3 = [taskInfo, thinking1, answer1]
    
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, determining relationship for planet #1, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], 
                                       'Review relationship determination for planet #1 and provide feedback.', 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining relationship for planet #1, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    print('Subtask 3 answer: ', sub_tasks[-1])

    cot_reflect_instruction_4 = 'Sub-task 4: Determine the relationship between the velocity of the star and the orbital period of the planet for planet #2 using the RV method.'
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    
    cot_inputs_4 = [taskInfo, thinking2, answer2]
    
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, determining relationship for planet #2, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], 
                                       'Review relationship determination for planet #2 and provide feedback.', 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining relationship for planet #2, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    
    print('Subtask 4 answer: ', sub_tasks[-1])

    debate_instruction_5 = 'Sub-task 5: Calculate the ratio of the orbital periods of planet #2 to planet #1 using the velocities derived from the wavelength shifts.'
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f'Debate agent {agent.id}, round {r}, calculating ratio, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 'Sub-task 5: Make final decision on the ratio of orbital periods.', 
                                                 is_sub_task=True)
    agents.append(f'Final Decision agent, calculating ratio, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    
    print('Subtask 5 answer: ', sub_tasks[-1])

    cot_instruction_6 = 'Sub-task 6: Select the correct choice from the given options that matches the calculated ratio of the orbital periods.'
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, selecting correct choice, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    
    print('Subtask 6 answer: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer