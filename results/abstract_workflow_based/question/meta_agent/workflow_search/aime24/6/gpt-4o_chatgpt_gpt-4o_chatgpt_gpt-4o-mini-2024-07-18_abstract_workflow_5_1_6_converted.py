async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Identify the equations and constraints for the rectangular boxes: surface area = 54 and volume = 23.
    cot_instruction = "Sub-task 1: Identify the equations and constraints for the rectangular boxes: surface area = 54 and volume = 23."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, determine equations and constraints, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Determine the relationship between the dimensions of the box (length, width, height) and the radius of the smallest containing sphere.
    cot_instruction_2 = "Sub-task 2: Based on the output of sub-task 1, determine the relationship between the dimensions of the box and the radius of the smallest containing sphere."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, determine relationship between dimensions and radius, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Calculate the possible dimensions of the rectangular boxes that satisfy both the surface area and volume constraints.
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1, calculate the possible dimensions of the rectangular boxes."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1]
    
    # The CoT agent generates the initial calculation.
    thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, calculating possible dimensions, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        # The critic agent evaluates the calculation and provides feedback for correction.
        feedback, correct = critic_agent_3([taskInfo, thinking3, answer3], "please review the possible dimensions calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # The CoT agent refines the calculation based on the critic's feedback.
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining possible dimensions, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Compute the radius of the smallest sphere that can contain a box with the calculated dimensions.
    cot_reflect_instruction_4 = "Sub-task 4: Based on the outputs from Sub-task 2 and Sub-task 3, compute the radius of the smallest sphere that can contain the box."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    
    # The CoT agent generates the initial calculation.
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, computing radius, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        # The critic agent evaluates the calculation and provides feedback for correction.
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], "please review the radius calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # The CoT agent refines the calculation based on the critic's feedback.
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining radius, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Calculate the value of r^2 and express it as a fraction in simplest form.
    debate_instruction_5 = "Sub-task 5: Based on the output of sub-task 4, calculate the value of r^2 and express it as a fraction in simplest form."
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.4) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        # In each round, agents propose their solutions, considering previous arguments.
        for i, agent in enumerate(debate_agents_5):
            input_infos_5 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos_5.extend(all_thinking5[r-1])
            thinking5, answer5 = agent(input_infos_5, debate_instruction_5, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating r^2, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
            
    # A final decision agent reviews the entire debate to produce a single, synthesized answer.
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on r^2.", is_sub_task=True)
    agents.append(f'Final Decision agent, for the purpose of calculating r^2, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 6: Find the sum of the numerator and denominator of the fraction representing r^2.
    cot_instruction_6 = "Sub-task 6: Based on the output of sub-task 5, find the sum of the numerator and denominator of the fraction representing r^2."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, calculating sum of numerator and denominator, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer