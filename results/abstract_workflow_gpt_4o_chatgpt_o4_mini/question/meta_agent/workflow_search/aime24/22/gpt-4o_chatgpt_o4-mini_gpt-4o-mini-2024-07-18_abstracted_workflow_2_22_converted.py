async def forward_22(self, taskInfo):
    from collections import Counter
    
    print('Task Requirement: ', taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Identify possible lists
    cot_instruction_1 = 'Sub-task 1: Identify possible lists of positive integers whose sum is 30.'
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, identifying lists, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])

    # Stage 0: Select lists with unique mode 9
    cot_sc_instruction_2 = 'Sub-task 2: From the lists identified in subtask_1, select those where the unique mode is 9.'
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                                model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, selecting lists with mode 9, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    
    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])

    # Stage 0: Select lists where median is not in the list
    cot_reflect_instruction_3 = 'Sub-task 3: From the lists identified in subtask_2, select those where the median is a positive integer not present in the list.'
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, selecting lists with median condition, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], 
                                       'please review the median condition and provide its limitations.', 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining median condition, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])

    # Stage 1: Verify the list
    cot_reflect_instruction_4 = 'Sub-task 4: Verify that the list from subtask_3 satisfies all the given conditions: sum is 30, unique mode is 9, and median is not in the list.'
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, verifying list, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], 
                                       'please review the list verification and provide its limitations.', 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining list verification, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])

    # Stage 2: Compute sum of squares
    cot_instruction_5 = 'Sub-task 5: Compute the sum of the squares of the elements in the verified list from subtask_4.'
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, computing sum of squares, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5: ', sub_tasks[-1])

    # Stage 3: Validate sum of squares
    cot_reflect_instruction_6 = 'Sub-task 6: Validate that the computed sum of squares is correct and consistent with the list properties.'
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, validating sum of squares, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], 
                                       'please review the sum of squares validation and provide its limitations.', 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining sum of squares validation, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer