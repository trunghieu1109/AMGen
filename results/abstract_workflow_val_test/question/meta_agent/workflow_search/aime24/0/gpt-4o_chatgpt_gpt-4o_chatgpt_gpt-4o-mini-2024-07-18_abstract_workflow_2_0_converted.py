async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning
    # Sub-task 1: Calculate walking time at s km/h
    cot_instruction_1 = "Sub-task 1: Calculate the walking time when Aya walks at s kilometers per hour, given that the total time is 4 hours including t minutes at the coffee shop."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculating walking time at s km/h, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Calculate walking time at s+2 km/h
    cot_instruction_2 = "Sub-task 2: Calculate the walking time when Aya walks at s+2 kilometers per hour, given that the total time is 2 hours and 24 minutes including t minutes at the coffee shop."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, calculating walking time at s+2 km/h, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Determine time t spent at the coffee shop
    cot_sc_instruction_3 = "Sub-task 3: Determine the time t spent at the coffee shop using the results from subtask_1 and subtask_2."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}

    for i in range(N):
        thinking3, answer3 = cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, determining coffee shop time t, thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3

    answer3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3]
    answer3 = answermapping_3[answer3]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Intermediate Inference and Validation
    # Sub-task 4: Calculate walking time at s+1/2 km/h
    cot_reflect_instruction_4 = "Sub-task 4: Calculate the walking time when Aya walks at s+1/2 kilometers per hour, using the distance of 9 kilometers."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round

    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, calculating walking time at s+1/2 km/h, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback4, correct4 = critic_agent_4([taskInfo, thinking4, answer4], "please review the walking time calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}')
        if correct4.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining walking time at s+1/2 km/h, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Final Output Generation and Integration
    # Sub-task 5: Calculate total time in minutes
    debate_instruction_5 = "Sub-task 5: Calculate the total time in minutes for Ayas walk and coffee shop visit when walking at s+1/2 kilometers per hour, using the walking time from subtask_4 and the coffee shop time t from subtask_3."
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round

    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for agent in debate_agents_5:
            input_infos_5 = [taskInfo, thinking4, answer4, thinking3, answer3]
            if r > 0:
                input_infos_5.extend(all_thinking5[r-1])
            thinking5, answer5 = agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating total time in minutes, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_answers_5 = [ans.content for ans in all_answer5[-1]]
    final_answercontent_5 = Counter(final_answers_5).most_common(1)[0][0]
    index_5 = final_answers_5.index(final_answercontent_5)
    thinking5 = all_thinking5[-1][index_5]
    answer5 = all_answer5[-1][index_5]
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer