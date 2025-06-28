async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    
    # Stage 1: Knowledge Extraction and Decomposition
    # Sub-task 1: Identify the symmetries of a regular octagon
    cot_instruction = "Sub-task 1: Identify the symmetries of a regular octagon, specifically the rotational symmetries."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, determine symmetries, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Determine the total number of ways to color the vertices
    cot_instruction = "Sub-task 2: Determine the total number of ways to color the vertices of the octagon with two colors."
    thinking2, answer2 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, determine total colorings, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Understand the condition for a valid rotation
    cot_instruction = "Sub-task 3: Understand the condition for a valid rotation: all blue vertices must end up at positions where there were originally red vertices."
    thinking3, answer3 = cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, understand valid rotation condition, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 2: Comprehensive Scenario Evaluation
    # Sub-task 4: Calculate the number of colorings that satisfy the condition for each rotation
    cot_instruction = "Sub-task 4: Calculate the number of colorings that satisfy the condition for each possible rotation of the octagon."
    thinking4, answer4 = cot_agent([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculate valid colorings for rotations, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Evaluate the probability of each valid rotation
    cot_instruction = "Sub-task 5: Evaluate the probability of each valid rotation occurring given the total number of colorings."
    thinking5, answer5 = cot_agent([taskInfo, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, evaluate probability of valid rotations, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])
    
    # Stage 3: Case Aggregation and Filtering
    # Sub-task 6: Aggregate the probabilities of all valid rotations
    cot_instruction = "Sub-task 6: Aggregate the probabilities of all valid rotations to find the total probability that the condition is satisfied."
    thinking6, answer6 = cot_agent([taskInfo, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, aggregate probabilities, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])
    
    # Stage 4: Intermediate Output Calculation
    # Sub-task 7: Calculate the intermediate output
    cot_instruction = "Sub-task 7: Calculate the intermediate output, which is the probability in the form of a fraction m/n."
    thinking7, answer7 = cot_agent([taskInfo, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculate intermediate output, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])
    
    # Stage 5: Final Answer Derivation
    # Sub-task 8: Convert the intermediate output to the specific format
    cot_instruction = "Sub-task 8: Convert the intermediate output to the specific format required by the problem, which is m+n."
    thinking8, answer8 = cot_agent([taskInfo, thinking7, answer7], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, convert to specific format, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print("Step 8: ", sub_tasks[-1])
    
    # Stage 6: Final Output Verification
    # Sub-task 9: Verify the correctness of the final output
    cot_instruction = "Sub-task 9: Verify the correctness of the final output by checking against the task requirements and intermediate outputs."
    thinking9, answer9 = cot_agent([taskInfo, thinking8, answer8], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, verify final output, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    print("Step 9: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer