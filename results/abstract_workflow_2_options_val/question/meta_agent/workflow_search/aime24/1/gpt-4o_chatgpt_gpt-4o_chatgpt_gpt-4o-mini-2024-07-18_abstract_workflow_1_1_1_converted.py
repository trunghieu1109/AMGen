async def forward(self, taskInfo):
    
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Analyze the geometric configuration of triangle ABC inscribed in circle ω
    cot_instruction = "Sub-task 1: Analyze the geometric configuration of triangle ABC inscribed in circle ω and the tangents at B and C intersecting at D."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing geometric configuration, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Determine the relationship between line AD and its intersection with circle ω at point P
    cot_instruction = "Sub-task 2: Determine the relationship between line AD and its intersection with circle ω at point P."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, determining relationship of AD and P, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Identify the properties of the triangle ABC
    cot_instruction = "Sub-task 3: Identify the properties of the triangle ABC, such as side lengths and their implications on the circle."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying properties of triangle ABC, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Use the Power of a Point theorem to find the length of AP
    cot_reflect_instruction = "Sub-task 4: Use the Power of a Point theorem to find the length of AP in terms of known quantities."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking2, answer2, thinking3, answer3]
    
    thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating length of AP, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking4, answer4], 
                                       "Critically evaluate the calculation of AP and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining length of AP, thinking: {thinking4.content}; answer: {answer4.content}')
    
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Simplify the expression for AP to the form m/n
    cot_instruction = "Sub-task 5: Simplify the expression for AP to the form m/n where m and n are relatively prime integers."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent([taskInfo, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, simplifying expression for AP, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 6: Calculate m + n from the simplified expression of AP
    cot_instruction = "Sub-task 6: Calculate m + n from the simplified expression of AP."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent([taskInfo, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating m + n, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer