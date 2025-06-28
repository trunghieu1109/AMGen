async def forward(self, taskInfo):
    from collections import Counter
    
    sub_tasks = []
    agents = []

    # Stage 1: Expression Analysis and Data Interpretation
    
    # Sub-task 1: Analyze log_x(y^x) = 10
    cot_instruction_1 = "Sub-task 1: Analyze the expression log_x(y^x) = 10 to find a relationship between x and y."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, analyzing log_x(y^x) = 10, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze log_y(x^(4y)) = 10
    cot_instruction_2 = "Sub-task 2: Based on Sub-task 1 output, analyze the expression log_y(x^(4y)) = 10 to find another relationship between x and y."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                                model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    
    for i in range(N):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, analyzing log_y(x^(4y)) = 10, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    
    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Intermediate Computation and Synthesis
    
    # Sub-task 3: Synthesize relationships
    cot_reflect_instruction_3 = "Sub-task 3: Synthesize the relationships found in Sub-task 1 and Sub-task 2 to express y in terms of x or vice versa."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, synthesizing relationships, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        feedback_3, correct_3 = critic_agent_3([taskInfo, thinking3, answer3], 
                                               "Critically evaluate the synthesis of relationships.", 
                                               i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback_3.content}; answer: {correct_3.content}')
        if correct_3.content == 'True':
            break
        cot_inputs_3.extend([thinking3, answer3, feedback_3])
        thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining synthesis, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compute values of x and y
    cot_reflect_instruction_4 = "Sub-task 4: Compute the values of x and y using the synthesized relationships and verify the consistency of the solution."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, computing values of x and y, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback_4, correct_4 = critic_agent_4([taskInfo, thinking4, answer4], 
                                               "Verify the consistency of the computed values.", 
                                               i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback_4.content}; answer: {correct_4.content}')
        if correct_4.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback_4])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining computation, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Final Answer Generation and Integration
    
    # Sub-task 5: Calculate xy
    cot_reflect_instruction_5 = "Sub-task 5: Calculate the product xy using the values of x and y obtained in stage 2."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, calculating xy, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback_5, correct_5 = critic_agent_5([taskInfo, thinking5, answer5], 
                                               "Review the calculation of xy for accuracy.", 
                                               i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback_5.content}; answer: {correct_5.content}')
        if correct_5.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback_5])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining calculation of xy, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer