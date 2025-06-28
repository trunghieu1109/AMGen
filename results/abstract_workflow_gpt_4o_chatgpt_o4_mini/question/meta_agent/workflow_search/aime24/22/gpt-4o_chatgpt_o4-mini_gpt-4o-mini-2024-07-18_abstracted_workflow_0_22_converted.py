async def forward_22(self, taskInfo):
    from collections import Counter
    
    print('Task Requirement: ', taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify lists of positive integers
    # Sub-task 1: Identify possible lists of positive integers that sum to 30 using CoT
    cot_instruction_1 = 'Sub-task 1: Identify possible lists of positive integers that sum to 30.'
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, identifying lists, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Subtask 1 answer: ', sub_tasks[-1])

    # Sub-task 2: Filter lists where the unique mode is 9 using Reflexion
    cot_reflect_instruction_2 = 'Sub-task 2: From the lists identified in subtask_1, filter those where the unique mode is 9.'
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                                  model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    
    cot_inputs_2 = [taskInfo, thinking1, answer1]
    
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_2.id}, filtering lists with mode 9, thinking: {thinking2.content}; answer: {answer2.content}')

    for i in range(N_max_2):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], 
                                       'Review and refine list filtering for mode 9.', 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_2.id}, refining list filtering, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Subtask 2 answer: ', sub_tasks[-1])

    # Sub-task 3: Filter lists where the median is not in the list using SC_CoT
    cot_sc_instruction_3 = 'Sub-task 3: From the lists identified in subtask_2, filter those where the median is a positive integer not present in the list.'
    N_3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                                model=self.node_model, temperature=0.5) for _ in range(N_3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    
    for i in range(N_3):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, filtering lists with median condition, thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    
    answer3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3]
    answer3 = answermapping_3[answer3]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Subtask 3 answer: ', sub_tasks[-1])

    # Stage 2: Calculate the sum of the squares of the items
    # Sub-task 4: Calculate the sum of the squares using CoT
    cot_instruction_4 = 'Sub-task 4: Calculate the sum of the squares of the items in the list(s) identified in subtask_3.'
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                               model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, calculating sum of squares, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Subtask 4 answer: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer