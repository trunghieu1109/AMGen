async def forward(self, taskInfo):
    from collections import Counter
    
    sub_tasks = []
    agents = []

    # Stage 1: Expression Analysis and Data Interpretation
    
    # Sub-task 1: Analyze the configuration of the eight circles of radius 34
    cot_instruction_1 = "Sub-task 1: Analyze the configuration of the eight circles of radius 34 and their tangency conditions."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, analyzing eight circles of radius 34, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    
    # Sub-task 2: Analyze the configuration of the 2024 circles of radius 1
    cot_instruction_2 = "Sub-task 2: Analyze the configuration of the 2024 circles of radius 1 and their tangency conditions."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, analyzing 2024 circles of radius 1, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    # Sub-task 3: Determine the relationship between the circles and the sides of triangle ABC
    cot_instruction_3 = "Sub-task 3: Determine the relationship between the circles and the sides of triangle ABC, focusing on the tangency conditions."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, determining relationships, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    # Stage 2: Intermediate Computation and Synthesis
    
    # Sub-task 4: Compute the inradius of triangle ABC
    cot_reflect_instruction_4 = "Sub-task 4: Compute the inradius of triangle ABC using the relationships and configurations analyzed in Stage 1."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, computing inradius, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], 
                                           "Critically evaluate the inradius calculation for correctness.", 
                                           i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining inradius, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    
    # Sub-task 5: Express the inradius in the form m/n
    cot_instruction_5 = "Sub-task 5: Express the inradius in the form m/n where m and n are relatively prime integers."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, expressing inradius, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    
    # Stage 3: Final Answer Generation and Integration
    
    # Sub-task 6: Calculate m+n
    cot_instruction_6 = "Sub-task 6: Calculate m+n from the expression of the inradius obtained in Stage 2."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, calculating m+n, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer