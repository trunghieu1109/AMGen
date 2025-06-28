async def forward(self, taskInfo):
    from collections import Counter
    import math
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Sub-task 1: Identify the condition for a number n to be b-beautiful
    cot_instruction = "Sub-task 1: Identify the condition for a number n to be b-beautiful, which involves having exactly two digits in base b and the sum of these digits equaling the square root of n."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying condition for b-beautiful, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine the range of numbers that can be represented with two digits in base b
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, determine the range of numbers that can be represented with two digits in base b."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, determining range for two-digit numbers, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate the possible values of n for each base b
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, calculate the possible values of n for each base b by iterating over all two-digit combinations in base b and checking if the sum of the digits equals the square root of n."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating possible values of n, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review possible values of n calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining possible values of n, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Count the number of b-beautiful numbers for each base b
    debate_instruction_4 = "Sub-task 4: Count the number of b-beautiful numbers for each base b and identify the smallest base b for which there are more than ten such numbers."
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, counting b-beautiful numbers, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the smallest base b.", is_sub_task=True)
    agents.append(f'Final Decision agent, determining smallest base b, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer