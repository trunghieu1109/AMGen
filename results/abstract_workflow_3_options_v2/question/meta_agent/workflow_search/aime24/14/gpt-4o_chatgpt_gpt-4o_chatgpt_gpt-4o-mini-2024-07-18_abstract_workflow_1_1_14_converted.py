async def forward(self, taskInfo):
   
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Analyze the hyperbola equation
    cot_instruction = "Sub-task 1: Analyze the equation x^2/20 - y^2/24 = 1 to understand its geometric properties and the range of possible points (A, B, C, D) on it."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing hyperbola equation, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Determine conditions for rhombus
    cot_sc_instruction = "Sub-task 2: Based on Sub-task 1 output, determine the conditions for points A, B, C, and D to form a rhombus with diagonals intersecting at the origin."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, determining rhombus conditions, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Compute expressions for diagonals
    cot_reflect_instruction = "Sub-task 3: Based on Sub-task 1 and Sub-task 2 outputs, compute expressions for diagonals AC and BD and use the condition that they intersect at the origin."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, computing diagonal expressions, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], 
                                       "Critically evaluate the diagonal expressions and provide limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining diagonal expressions, thinking: {thinking3.content}; answer: {answer3.content}')
    
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Determine maximum value of BD^2
    cot_reflect_instruction = "Sub-task 4: Determine the maximum possible value of BD^2 by analyzing the constraints derived from the rhombus and hyperbola properties."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, determining maximum BD^2, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking4, answer4], 
                                       "Review maximum BD^2 calculation for accuracy and provide limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining maximum BD^2, thinking: {thinking4.content}; answer: {answer4.content}')
    
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Integrate analyses to find greatest real number less than BD^2
    cot_reflect_instruction = "Sub-task 5: Integrate outputs from all previous subtasks to find the greatest real number less than BD^2 for all such rhombi."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    
    thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, integrating analyses, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking5, answer5], 
                                       "Review final integration for completeness and provide limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining final integration, thinking: {thinking5.content}; answer: {answer5.content}')
    
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer