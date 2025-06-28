async def forward(self, taskInfo):
    from collections import Counter
    
    sub_tasks = []
    agents = []

    # Sub-task 1: Identify the equation for the total time taken when Aya walks at speed s, including time spent in the coffee shop.
    cot_instruction_1 = "Sub-task 1: Identify the equation for the total time taken when Aya walks at speed s, including time spent in the coffee shop."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, identifying equation for speed s, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Identify the equation for the total time taken when Aya walks at speed s+2, including time spent in the coffee shop.
    cot_instruction_2 = "Sub-task 2: Identify the equation for the total time taken when Aya walks at speed s+2, including time spent in the coffee shop."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, identifying equation for speed s+2, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Solve the system of equations from subtask_1 and subtask_2 to find the values of s and t.
    cot_sc_instruction_3 = "Sub-task 3: Solve the system of equations from subtask_1 and subtask_2 to find the values of s and t."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.7) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, solving for s and t, thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3]
    answer3 = answermapping_3[answer3]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Sub-task 4: Calculate the time taken for Aya to walk 9 kilometers at speed s+0.5.
    cot_instruction_4 = "Sub-task 4: Calculate the time taken for Aya to walk 9 kilometers at speed s+0.5."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, calculating time for speed s+0.5, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Add the time spent in the coffee shop to the time calculated in subtask_4 to find the total time.
    cot_reflect_instruction_5 = "Sub-task 5: Add the time spent in the coffee shop to the time calculated in subtask_4 to find the total time."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking3, answer3, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, calculating total time, thinking: {thinking5.content}; answer: {answer5.content}')
    for i in range(N_max_5):
        feedback_5, correct_5 = critic_agent_5([taskInfo, thinking5, answer5], "please review total time calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback_5.content}; answer: {correct_5.content}')
        if correct_5.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback_5])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining total time, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer