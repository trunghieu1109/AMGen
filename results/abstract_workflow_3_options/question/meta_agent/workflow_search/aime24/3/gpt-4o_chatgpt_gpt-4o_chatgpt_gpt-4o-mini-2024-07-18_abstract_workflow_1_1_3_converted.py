async def forward(self, taskInfo):
    from collections import Counter
    
    sub_tasks = []
    agents = []

    # Stage 1: Expression Analysis and Data Interpretation
    
    # Sub-task 1: Analyze f(x) = ||x| - 1/2|
    cot_instruction_1 = "Sub-task 1: Analyze f(x) = ||x| - 1/2|, determining its behavior, range, and key characteristics with context from taskInfo"
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, analyzing f(x), thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze g(x) = ||x| - 1/4|
    cot_instruction_2 = "Sub-task 2: Based on Sub-task 1 output, analyze g(x) = ||x| - 1/4| and its relationship to f(x)"
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                                model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    
    for i in range(N):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, analyzing g(x), thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    
    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze f(sin(2πx)) and f(cos(3πy))
    cot_instruction_3 = "Sub-task 3: Analyze the composition f(sin(2πx)) and f(cos(3πy)) to understand their behavior and range"
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, analyzing compositions, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze 4g(f(sin(2πx))) and 4g(f(cos(3πy)))
    cot_instruction_4 = "Sub-task 4: Analyze the function 4g(f(sin(2πx))) and 4g(f(cos(3πy))) to determine their behavior and range"
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, analyzing scaled functions, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Intermediate Computation and Synthesis
    
    # Sub-task 5: Determine conditions for intersection
    cot_reflect_instruction_5 = "Sub-task 5: Determine the conditions for intersection of the graphs y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy)))"
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, determining intersection conditions, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback5, correct5 = critic_agent_5([taskInfo, thinking5, answer5], 
                                             "Critically evaluate the intersection conditions and provide its limitations.", 
                                             i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback5.content}; answer: {correct5.content}')
        if correct5.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining intersection conditions, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compute number of solutions
    cot_instruction_6 = "Sub-task 6: Compute the number of solutions to the intersection conditions derived in subtask 5"
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, computing number of solutions, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Stage 3: Final Answer Generation and Integration
    
    # Sub-task 7: Integrate analyses to determine number of intersections
    cot_reflect_instruction_7 = "Sub-task 7: Integrate all previous analyses to determine the number of intersections of the graphs"
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    thinking7, answer7 = cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_7.id}, determining number of intersections, thinking: {thinking7.content}; answer: {answer7.content}')

    for i in range(N_max_7):
        feedback7, correct7 = critic_agent_7([taskInfo, thinking7, answer7], 
                                             "Review final answer for completeness, accuracy, and alignment with original query", 
                                             i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}')
        if correct7.content == 'True':
            break
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_7.id}, refining number of intersections, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer