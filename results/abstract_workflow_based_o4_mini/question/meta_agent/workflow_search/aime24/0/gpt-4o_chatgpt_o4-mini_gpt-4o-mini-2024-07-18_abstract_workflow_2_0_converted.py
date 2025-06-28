async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    cot_instruction_1 = "Sub-task 1: Calculate Aya's walking speed s using the condition that at speed s she takes 4 hours, including t minutes in the coffee shop. Given that the total walk is 9 kilometers in 4 hours minus the coffee break time."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculating walking speed s, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, calculate the coffee shop time t using the condition that when she speeds up by 2 km/h, the journey is 2 hours and 24 minutes including t minutes."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, calculating coffee shop time t, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_reflect_instruction_3 = "Sub-task 3: Calculate how long the entire journey takes when she walks at s + 2 km/h, including the coffee shop time, ensuring the total is 2 hours and 24 minutes. Verify by leveraging prior calculations and ensuring coherence."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, calculating journey time at s+2 km/h, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        feedback, correct = critic_agent_3([taskInfo, thinking3, answer3], "please review the journey time calculation at s+2 km/h and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining journey time at s+2 km/h, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Subtask 3 answer: ", sub_tasks[-1])

    debate_instruction_5 = "Sub-task 5: Calculate the time spent on the walk at s + 0.5 km/h, leveraging the coffee shop time calculated in sub-task 2 and considering the evolved speed."
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round

    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for agent in debate_agents_5:
            input_infos_5 = [taskInfo, thinking3, answer3] if r == 0 else [taskInfo, thinking3, answer3] + all_thinking5[r - 1] + all_answer5[r - 1]
            thinking5, answer5 = agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating time at s+0.5 km/h, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make a final decision on the journey time at s+0.5 km/h.", is_sub_task=True)
    agents.append(f'Final Decision agent, calculating journey time at s+0.5 km/h, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Subtask 5 answer: ", sub_tasks[-1])

    subtask_6_instruction = "Sub-task 6: Integrate results from previous stages to determine total journey time with speed s+0.5 km/h in minutes."
    integrated_thinking_6 = LLMAgentBase(['thinking', 'answer'], 'Integrated Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = integrated_thinking_6([taskInfo, thinking5, answer5], subtask_6_instruction, is_sub_task=True)
    agents.append(f'Integrated agent {integrated_thinking_6.id}, integrating results, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer