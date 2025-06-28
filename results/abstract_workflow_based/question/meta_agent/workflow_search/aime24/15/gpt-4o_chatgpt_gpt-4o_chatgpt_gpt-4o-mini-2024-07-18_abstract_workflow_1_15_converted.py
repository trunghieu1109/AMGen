async def forward(self, taskInfo):
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Calculate the total number of residents who own at least one of the three items.
    cot_instruction = "Sub-task 1: Calculate the total number of residents who own at least one of the three items: diamond ring, golf clubs, or garden spade."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating total owners of at least one item, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Calculate the number of residents who own exactly one of the three items.
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, calculate the number of residents who own exactly one of the three items."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, determining exact owners of one item, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate the number of residents who own at least two of the three items.
    cot_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, calculate the number of residents who own at least two of the three items."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating owners of at least two items, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Calculate the number of residents who own exactly three of the three items.
    cot_instruction = "Sub-task 4: Based on the output from Sub-task 3, calculate the number of residents who own exactly three of the three items."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent([taskInfo, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating exact owners of three items, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 5: Calculate the number of residents who own all four items.
    cot_reflect_instruction = "Sub-task 5: Based on the outputs from Sub-task 4, calculate the number of residents who own all four items."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking4, answer4]
    thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating owners of all four items, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking5, answer5], "please review the calculation of owners of all four items and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining owners of all four items, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer