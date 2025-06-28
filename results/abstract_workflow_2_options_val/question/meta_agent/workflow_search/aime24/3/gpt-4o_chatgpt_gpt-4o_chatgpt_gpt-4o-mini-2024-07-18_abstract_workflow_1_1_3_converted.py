async def forward(self, taskInfo):
   
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Analyze the function f(x) = ||x| - 1/2|
    cot_instruction = "Sub-task 1: Analyze the function f(x) = ||x| - 1/2| to determine its behavior, range, and characteristics."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing f(x), thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Analyze the function g(x) = ||x| - 1/4|
    cot_instruction = "Sub-task 2: Analyze the function g(x) = ||x| - 1/4| to determine its behavior, range, and characteristics."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing g(x), thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Analyze the composition f(sin(2πx))
    cot_instruction = "Sub-task 3: Analyze the composition f(sin(2πx)) and determine its behavior and range."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing f(sin(2πx)), thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Sub-task 4: Analyze the composition f(cos(3πy))
    cot_instruction = "Sub-task 4: Analyze the composition f(cos(3πy)) and determine its behavior and range."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing f(cos(3πy)), thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Analyze the function 4g(x)
    cot_instruction = "Sub-task 5: Analyze the function 4g(x) and determine its behavior and range."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing 4g(x), thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 6: Compute the expression 4g(f(sin(2πx)))
    cot_reflect_instruction = "Sub-task 6: Compute the expression 4g(f(sin(2πx))) and determine its behavior and range."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking3, answer3, thinking5, answer5]
    
    thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, computing 4g(f(sin(2πx))), thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking6, answer6], 
                                       "Critically evaluate the computation of 4g(f(sin(2πx))) and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining 4g(f(sin(2πx))), thinking: {thinking6.content}; answer: {answer6.content}')
    
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    # Sub-task 7: Compute the expression 4g(f(cos(3πy)))
    cot_reflect_instruction = "Sub-task 7: Compute the expression 4g(f(cos(3πy))) and determine its behavior and range."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking4, answer4, thinking5, answer5]
    
    thinking7, answer7 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, computing 4g(f(cos(3πy))), thinking: {thinking7.content}; answer: {answer7.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking7, answer7], 
                                       "Critically evaluate the computation of 4g(f(cos(3πy))) and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking7, answer7, feedback])
        thinking7, answer7 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining 4g(f(cos(3πy))), thinking: {thinking7.content}; answer: {answer7.content}')
    
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 8: Determine the number of intersections
    cot_reflect_instruction = "Sub-task 8: Determine the number of intersections of the graphs y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy)))."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking6, answer6, thinking7, answer7]
    
    thinking8, answer8 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, determining intersections, thinking: {thinking8.content}; answer: {answer8.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking8, answer8], 
                                       "Review the determination of intersections and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking8, answer8, feedback])
        thinking8, answer8 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining intersections determination, thinking: {thinking8.content}; answer: {answer8.content}')
    
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
