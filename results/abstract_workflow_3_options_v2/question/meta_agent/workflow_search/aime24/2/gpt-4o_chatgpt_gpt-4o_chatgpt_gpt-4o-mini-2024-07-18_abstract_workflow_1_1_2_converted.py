async def forward(self, taskInfo):
    
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Analyze the probability of each vertex being independently colored
    cot_instruction = "Sub-task 1: Analyze the probability of each vertex being independently colored either red or blue with equal probability."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing vertex coloring probability, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Determine the rotational symmetries of a regular octagon
    cot_sc_instruction = "Sub-task 2: Based on Sub-task 1 output, determine the rotational symmetries of a regular octagon and how these symmetries affect the coloring configurations."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, determining rotational symmetries, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Identify the condition for blue vertices to rotate to red positions
    cot_instruction = "Sub-task 3: Identify the condition where all blue vertices can be rotated to positions originally occupied by red vertices."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying condition for rotation, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Compute the total number of possible colorings of the octagon
    cot_instruction = "Sub-task 4: Compute the total number of possible colorings of the octagon."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, computing total colorings, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Calculate the number of colorings that satisfy the condition
    cot_instruction = "Sub-task 5: Calculate the number of colorings that satisfy the condition identified in subtask 3."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating valid colorings, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 6: Determine the probability and express it in the form m/n
    cot_reflect_instruction = "Sub-task 6: Determine the probability that a randomly colored octagon can be rotated to satisfy the condition, and express it in the form m/n."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking5, answer5, thinking4, answer4]
    
    thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, determining probability, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking6, answer6], 
                                       "Review probability calculation for correctness and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining probability, thinking: {thinking6.content}; answer: {answer6.content}')
    
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    # Sub-task 7: Find the sum m+n
    cot_instruction = "Sub-task 7: Find the sum m+n where m and n are relatively prime positive integers from the probability m/n."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = cot_agent([taskInfo, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, finding sum m+n, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer