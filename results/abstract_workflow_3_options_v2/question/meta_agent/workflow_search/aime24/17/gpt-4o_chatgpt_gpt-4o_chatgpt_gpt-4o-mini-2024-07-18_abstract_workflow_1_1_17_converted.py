async def forward(self, taskInfo):
   
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Analyze the linear equation a + b + c = 300
    cot_instruction = "Sub-task 1: Analyze the linear equation a + b + c = 300 to understand its constraints on the variables a, b, and c."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing linear equation, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Analyze the polynomial equation
    cot_sc_instruction = "Sub-task 2: Based on Sub-task 1 output, analyze the polynomial equation a^2b + a^2c + b^2a + b^2c + c^2a + c^2b = 6000000 to understand its constraints and behavior."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, analyzing polynomial equation, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    # Sub-task 3: Identify the relationship between the linear and polynomial equations
    cot_reflect_instruction = "Sub-task 3: Identify the relationship between the linear and polynomial equations and how they jointly constrain the variables a, b, and c."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, identifying relationship, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], 
                                       "Critically evaluate the relationship between the equations and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining relationship, thinking: {thinking3.content}; answer: {answer3.content}')
    
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Compute potential values for a, b, and c
    cot_instruction = "Sub-task 4: Compute potential values or ranges for a, b, and c that satisfy the linear equation."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, computing potential values, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Evaluate the polynomial equation for potential values
    cot_sc_instruction = "Sub-task 5: Evaluate the polynomial equation for the potential values of a, b, and c obtained from the linear equation to check for consistency."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking5, answer5 = cot_agents[i]([taskInfo, thinking2, answer2, thinking4, answer4], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, evaluating polynomial equation, thinking: {thinking5.content}; answer: {answer5.content}')
        possible_answers.append(answer5.content)
        thinkingmapping[answer5.content] = thinking5
        answermapping[answer5.content] = answer5
    
    answer5 = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinkingmapping[answer5]
    answer5 = answermapping[answer5]
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    
    # Sub-task 6: Synthesize results to determine valid triples
    cot_reflect_instruction = "Sub-task 6: Synthesize the results from the evaluations to determine valid triples (a, b, c) that satisfy both equations."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking3, answer3, thinking5, answer5]
    
    thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, synthesizing results, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking6, answer6], 
                                       "Review the synthesis of results and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining synthesis, thinking: {thinking6.content}; answer: {answer6.content}')
    
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 7: Count the number of valid triples
    cot_instruction = "Sub-task 7: Count the number of valid triples (a, b, c) that satisfy both the linear and polynomial equations."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = cot_agent([taskInfo, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, counting valid triples, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer