async def forward_163(self, taskInfo):
    from collections import Counter
    
    print('Task Requirement: ', taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Calculate Mass Functions
    # Sub-task 1: Calculate mass function for system_1 using CoT
    cot_instruction_1 = 'Sub-task 1: Calculate the mass function for system_1 using the given radial velocity amplitudes (10 km/s and 5 km/s) and the orbital period (2 years).'
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculating mass function for system_1, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    
    print('Subtask 1 answer: ', sub_tasks[-1])

    # Sub-task 2: Calculate mass function for system_2 using CoT
    cot_instruction_2 = 'Sub-task 2: Calculate the mass function for system_2 using the given radial velocity amplitudes (15 km/s and 10 km/s) and the orbital period (1 year).'
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, calculating mass function for system_2, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    print('Subtask 2 answer: ', sub_tasks[-1])

    # Stage 2: Derive Total Masses
    # Sub-task 3: Derive total mass of system_1 using Reflexion
    cot_reflect_instruction_3 = 'Sub-task 3: Derive the total mass of system_1 using the mass function calculated in subtask_1.'
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    
    cot_inputs_3 = [taskInfo, thinking1, answer1]
    
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, deriving total mass of system_1, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], 
                                       'Review total mass derivation for system_1 and provide feedback.', 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining total mass of system_1, thinking: {thinking3.content}; answer: {answer3.content}')
    
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    print('Subtask 3 answer: ', sub_tasks[-1])

    # Sub-task 4: Derive total mass of system_2 using Reflexion
    cot_reflect_instruction_4 = 'Sub-task 4: Derive the total mass of system_2 using the mass function calculated in subtask_2.'
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    
    cot_inputs_4 = [taskInfo, thinking2, answer2]
    
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, deriving total mass of system_2, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], 
                                       'Review total mass derivation for system_2 and provide feedback.', 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining total mass of system_2, thinking: {thinking4.content}; answer: {answer4.content}')
    
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    
    print('Subtask 4 answer: ', sub_tasks[-1])

    # Stage 3: Compare Total Masses
    # Sub-task 5: Compare total masses using Debate
    debate_instruction_5 = 'Sub-task 5: Compare the total masses of system_1 and system_2 derived in subtasks 3 and 4 to determine the factor by which system_1 is more massive than system_2.'
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f'Debate agent {agent.id}, round {r}, comparing total masses, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking_5[r].append(thinking5)
            all_answer_5[r].append(answer5)
    
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], 
                                                 'Sub-task 5: Make final decision on the factor by which system_1 is more massive than system_2.', 
                                                 is_sub_task=True)
    agents.append(f'Final Decision agent, determining mass factor, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    
    print('Subtask 5 answer: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer