async def forward(self, taskInfo):
    
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Expression Analysis and Data Interpretation
    
    # Sub-task 1: Analyze f(x) = ||x| - 1/2|
    cot_instruction_1 = "Sub-task 1: Analyze f(x) = ||x| - 1/2|, determining its behavior, range, and key characteristics."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, analyzing f(x), thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze g(x) = ||x| - 1/4| with self-consistency
    cot_sc_instruction_2 = "Sub-task 2: Based on Sub-task 1 output, analyze g(x) = ||x| - 1/4| and its relationship to f(x)."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    
    for i in range(N):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, analyzing g(x), thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    
    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze f(sin(2πx)) and g(f(sin(2πx)))
    cot_instruction_3 = "Sub-task 3: Analyze the composition f(sin(2πx)) and g(f(sin(2πx))) to understand their behavior and range."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, analyzing f(sin(2πx)) and g(f(sin(2πx))), thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze f(cos(3πy)) and g(f(cos(3πy)))
    cot_instruction_4 = "Sub-task 4: Analyze the composition f(cos(3πy)) and g(f(cos(3πy))) to understand their behavior and range."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, analyzing f(cos(3πy)) and g(f(cos(3πy))), thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Intermediate Computation and Synthesis
    
    # Sub-task 5: Compute 4g(f(sin(2πx)))
    cot_reflect_instruction_5 = "Sub-task 5: Compute the expression 4g(f(sin(2πx))) and analyze its graph."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    
    cot_inputs_5 = [taskInfo, thinking3, answer3]
    
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, computing 4g(f(sin(2πx))), thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback5, correct5 = critic_agent_5([taskInfo, thinking5, answer5], 
                                       "Critically evaluate the graph of 4g(f(sin(2πx))) and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback5.content}; answer: {correct5.content}')
        if correct5.content == 'True':
            break
        
        cot_inputs_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining 4g(f(sin(2πx))), thinking: {thinking5.content}; answer: {answer5.content}')
    
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compute 4g(f(cos(3πy)))
    cot_reflect_instruction_6 = "Sub-task 6: Compute the expression 4g(f(cos(3πy))) and analyze its graph."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    
    cot_inputs_6 = [taskInfo, thinking4, answer4]
    
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, computing 4g(f(cos(3πy))), thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max_6):
        feedback6, correct6 = critic_agent_6([taskInfo, thinking6, answer6], 
                                       "Critically evaluate the graph of 4g(f(cos(3πy))) and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}')
        if correct6.content == 'True':
            break
        
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining 4g(f(cos(3πy))), thinking: {thinking6.content}; answer: {answer6.content}')
    
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Determine intersection conditions
    cot_reflect_instruction_7 = "Sub-task 7: Determine the conditions for intersection of the graphs y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy)))."
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    
    cot_inputs_7 = [taskInfo, thinking5, answer5, thinking6, answer6]
    
    thinking7, answer7 = cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_7.id}, determining intersection conditions, thinking: {thinking7.content}; answer: {answer7.content}')

    for i in range(N_max_7):
        feedback7, correct7 = critic_agent_7([taskInfo, thinking7, answer7], 
                                       "Review intersection conditions for completeness and accuracy.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}')
        if correct7.content == 'True':
            break
        
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_7.id}, refining intersection conditions, thinking: {thinking7.content}; answer: {answer7.content}')
    
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    # Stage 3: Final Answer Generation and Integration
    
    # Sub-task 8: Calculate number of intersection points
    cot_reflect_instruction_8 = "Sub-task 8: Calculate the number of intersection points based on the conditions derived in subtask_7."
    cot_agent_8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    
    cot_inputs_8 = [taskInfo, thinking7, answer7]
    
    thinking8, answer8 = cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_8.id}, calculating number of intersection points, thinking: {thinking8.content}; answer: {answer8.content}')

    for i in range(N_max_8):
        feedback8, correct8 = critic_agent_8([taskInfo, thinking8, answer8], 
                                       "Review calculation of intersection points for accuracy.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback8.content}; answer: {correct8.content}')
        if correct8.content == 'True':
            break
        
        cot_inputs_8.extend([thinking8, answer8, feedback8])
        thinking8, answer8 = cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_8.id}, refining calculation of intersection points, thinking: {thinking8.content}; answer: {answer8.content}')
    
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer