async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Sub-task 1: Solve the equation log base x of y raised to the power of x equals 10 to express y in terms of x.
    cot_instruction_1 = "Sub-task 1: Solve the equation log base x of y raised to the power of x equals 10 to express y in terms of x."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, solving for y in terms of x, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Solve the equation log base y of x raised to the power of 4y equals 10 to express x in terms of y.
    cot_instruction_2 = "Sub-task 2: Solve the equation log base y of x raised to the power of 4y equals 10 to express x in terms of y."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, solving for x in terms of y, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Use the expressions from subtask 1 and subtask 2 to find a consistent relationship between x and y.
    cot_reflect_instruction_3 = "Sub-task 3: Use the expressions from subtask 1 and subtask 2 to find a consistent relationship between x and y."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, finding relationship between x and y, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        feedback, correct = critic_agent_3([taskInfo, thinking3, answer3], "please review the relationship between x and y and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining relationship between x and y, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Sub-task 4: Calculate the product xy using the relationship found in subtask 3.
    debate_instruction_4 = "Sub-task 4: Calculate the product xy using the relationship found in subtask 3."
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.7) for role in self.debate_role]
    N_max_4 = self.max_round

    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for agent in debate_agents_4:
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating product xy, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)

    final_answers = [ans.content for ans in all_answer4[-1]]
    final_answercontent = Counter(final_answers).most_common(1)[0][0]
    index = final_answers.index(final_answercontent)
    thinking4 = all_thinking4[-1][index]
    answer4 = all_answer4[-1][index]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer