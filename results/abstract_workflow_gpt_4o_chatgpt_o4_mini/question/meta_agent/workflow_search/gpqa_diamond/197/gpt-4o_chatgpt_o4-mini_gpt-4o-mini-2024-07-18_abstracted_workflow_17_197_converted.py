async def forward_197(self, taskInfo):
    from collections import Counter
    
    print('Task Requirement: ', taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction = 'Sub-task 1: Calculate the concentration of Co(II) ions in the solution using the initial concentration c(Co) = 10^-2 M.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating Co(II) concentration, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 2: Calculate the concentration of SCN- ions in the solution using the initial concentration [SCN-] = 0.1 M.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating SCN- concentration, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 3: Apply the stability constant β1 to calculate the concentration of the Co(SCN)+ complex.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating Co(SCN)+ concentration, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 4: Apply the stability constant β2 to calculate the concentration of the Co(SCN)2 complex.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating Co(SCN)2 concentration, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 5: Apply the stability constant β3 to calculate the concentration of the Co(SCN)3- complex.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating Co(SCN)3- concentration, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 6: Apply the stability constant β4 to calculate the concentration of the Co(SCN)4^2- complex.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating Co(SCN)4^2- concentration, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 7: Sum the concentrations of all cobalt-containing species to find the total concentration of cobalt species in the solution.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent([taskInfo, thinking1, answer1, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, summing cobalt species concentrations, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Step 7: ', sub_tasks[-1])

    cot_instruction = 'Sub-task 8: Calculate the percentage of the Co(SCN)2 complex among all cobalt-containing species using the concentration from subtask_4 and the total concentration from subtask_7.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent([taskInfo, thinking4, answer4, thinking7, answer7], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating percentage of Co(SCN)2, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print('Step 8: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer