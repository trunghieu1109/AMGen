async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning
    
    # Sub-task 1: Calculate Aya's walking speed s when the walk takes 4 hours, including t minutes spent in the coffee shop.
    cot_instruction_1 = "Sub-task 1: Calculate Aya's walking speed s when the walk takes 4 hours, including t minutes spent in the coffee shop."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculating walking speed s, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Calculate Aya's walking speed s+2 when the walk takes 2 hours and 24 minutes, including t minutes spent in the coffee shop.
    cot_instruction_2 = "Sub-task 2: Calculate Aya's walking speed s+2 when the walk takes 2 hours and 24 minutes, including t minutes spent in the coffee shop."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, calculating walking speed s+2, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Set up equations based on the conditions from subtask 1 and subtask 2 to solve for s and t.
    cot_instruction_3 = "Sub-task 3: Set up equations based on the conditions from subtask 1 and subtask 2 to solve for s and t."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, setting up equations, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Stage 2: Intermediate Inference and Validation
    
    # Sub-task 4: Solve the equations to find the values of s and t.
    cot_reflect_instruction_4 = "Sub-task 4: Solve the equations to find the values of s and t."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round

    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, solving equations, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], "please review the solution of equations and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining solution, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Calculate the time it takes for Aya to walk 9 kilometers at s+1/2 kilometers per hour, including t minutes spent in the coffee shop.
    cot_instruction_5 = "Sub-task 5: Calculate the time it takes for Aya to walk 9 kilometers at s+1/2 kilometers per hour, including t minutes spent in the coffee shop."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, calculating time for s+1/2 speed, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # Stage 3: Final Output Generation and Integration
    
    # Sub-task 6: Convert the total time from hours and minutes to just minutes for the final answer.
    debate_instruction_6 = "Sub-task 6: Convert the total time from hours and minutes to just minutes for the final answer."
    debate_agents_6 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.7) for role in self.debate_role]
    N_max_6 = self.max_round

    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for agent in debate_agents_6:
            input_infos_6 = [taskInfo, thinking5, answer5]
            
            if r > 0:
                input_infos_6.extend(all_thinking6[r-1])
                
            thinking6, answer6 = agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, converting time to minutes, thinking: {thinking6.content}; answer: {answer6.content}')
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_answers_6 = [ans.content for ans in all_answer6[-1]]
    final_answercontent_6 = Counter(final_answers_6).most_common(1)[0][0]
    index_6 = final_answers_6.index(final_answercontent_6)
    thinking6 = all_thinking6[-1][index_6]
    answer6 = all_answer6[-1][index_6]
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer