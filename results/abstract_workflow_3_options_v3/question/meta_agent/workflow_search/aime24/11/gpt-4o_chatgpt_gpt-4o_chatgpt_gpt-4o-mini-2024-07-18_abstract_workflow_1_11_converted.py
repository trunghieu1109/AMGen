async def forward(self, taskInfo):
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Identify the total number of steps required to move from the lower left corner to the upper right corner on an 8x8 grid.
    cot_instruction_1 = "Sub-task 1: Calculate the total number of steps required to move from the lower left corner to the upper right corner on an 8x8 grid."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculating total steps, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine the number of direction changes required and how they can be distributed along the path.
    cot_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, determine the number of direction changes required and how they can be distributed along the path."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, determining direction changes, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate the number of ways to arrange the steps such that there are exactly four direction changes.
    cot_instruction_3 = "Sub-task 3: Calculate the number of ways to arrange the steps such that there are exactly four direction changes."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, arranging steps for direction changes, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Use the results from the previous subtasks to calculate the total number of valid paths that meet the criteria.
    cot_reflect_instruction_4 = "Sub-task 4: Based on the outputs from Sub-task 1, 2, and 3, calculate the total number of valid paths that meet the criteria."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    
    # Input for cot-agent
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    # generate the first answer.
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, calculating total valid paths, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        # use critic-agent to reflect and criticise the proposed answer from cot-agent
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], "please review total valid paths calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        
        # stop if the answer is correct
        if correct.content == 'True':
            break
        
        # merge recently answer and feedback from critic and then pass to cot-agent in the next round
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining total valid paths, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Verify the solution by considering edge cases and ensuring all conditions are met.
    cot_reflect_instruction_5 = "Sub-task 5: Verify the solution by considering edge cases and ensuring all conditions are met."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    
    # Input for cot-agent
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    
    # generate the first answer.
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, verifying solution, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        # use critic-agent to reflect and criticise the proposed answer from cot-agent
        feedback, correct = critic_agent_5([taskInfo, thinking5, answer5], "please review solution verification and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        
        # stop if the answer is correct
        if correct.content == 'True':
            break
        
        # merge recently answer and feedback from critic and then pass to cot-agent in the next round
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining solution verification, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer