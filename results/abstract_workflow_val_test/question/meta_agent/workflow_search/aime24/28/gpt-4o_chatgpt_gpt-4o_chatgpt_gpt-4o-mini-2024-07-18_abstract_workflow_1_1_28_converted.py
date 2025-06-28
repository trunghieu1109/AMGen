async def forward(self, taskInfo):
    
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Expression Analysis and Data Interpretation
    
    # Sub-task 1: Analyze the geometric properties of the torus
    cot_instruction_1 = "Sub-task 1: Analyze the geometric properties of the torus, including its major and minor radii."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, analyzing torus properties, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze the geometric properties of the sphere
    cot_instruction_2 = "Sub-task 2: Analyze the geometric properties of the sphere, particularly its radius."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, analyzing sphere properties, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Determine conditions for internal tangency
    cot_instruction_3 = "Sub-task 3: Determine the conditions for tangency between the torus and the sphere, focusing on the internal tangency."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, determining internal tangency, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine conditions for external tangency
    cot_instruction_4 = "Sub-task 4: Determine the conditions for tangency between the torus and the sphere, focusing on the external tangency."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, determining external tangency, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Intermediate Computation and Synthesis
    
    # Sub-task 5: Compute the radius of the circle of internal tangency (r_i)
    cot_reflect_instruction_5 = "Sub-task 5: Compute the radius of the circle of internal tangency (r_i) using the conditions derived in subtask_3."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    
    cot_inputs_5 = [taskInfo, thinking3, answer3]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, calculating r_i, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback, correct = critic_agent_5([taskInfo, thinking5, answer5], 
                                       "Critically evaluate the calculation of r_i and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining r_i, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compute the radius of the circle of external tangency (r_o)
    cot_reflect_instruction_6 = "Sub-task 6: Compute the radius of the circle of external tangency (r_o) using the conditions derived in subtask_4."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    
    cot_inputs_6 = [taskInfo, thinking4, answer4]
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, calculating r_o, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max_6):
        feedback, correct = critic_agent_6([taskInfo, thinking6, answer6], 
                                       "Critically evaluate the calculation of r_o and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining r_o, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Calculate the difference r_i - r_o and express it in the form of a fraction m/n
    cot_instruction_7 = "Sub-task 7: Calculate the difference r_i - r_o and express it in the form of a fraction m/n."
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = cot_agent_7([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_7.id}, calculating r_i - r_o, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    # Stage 3: Final Answer Generation and Integration
    
    # Sub-task 8: Find the sum m + n
    cot_instruction_8 = "Sub-task 8: Find the sum m + n, where m and n are relatively prime positive integers from the fraction obtained in subtask_7."
    cot_agent_8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking8, answer8 = cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_8.id}, finding m + n, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer