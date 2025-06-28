async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Subtask 1: Determine Aya's walking speed s using CoT.
    cot_instruction_1 = "Sub-task 1: Determine Aya's walking speed s using the information from the first scenario where she walks 9 kilometers and spends a total of 4 hours including the time t at the coffee shop."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating walking speed, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Subtask 2: Determine time t Aya spends at the coffee shop using SC-CoT.
    cot_sc_instruction_2 = "Sub-task 2: Determine the time t Aya spends at the coffee shop using the information from the second scenario where she walks at s + 2 kilometers per hour and has a total time of 2 hours and 24 minutes."
    N_2 = self.max_sc
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i, agent in enumerate(cot_sc_agents_2):
        thinking2, answer2 = agent([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, determining coffee shop time, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answerstr_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answerstr_2]
    answer2 = answermapping_2[most_common_answerstr_2]
    sub_tasks.append(f"Sub-task 2 output (most common): thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Subtask 3: Calculate the actual walking time for the first scenario using Debate pattern.
    debate_instruction_3 = "Sub-task 3: Calculate the actual walking time in the first scenario, by subtracting t (coffee shop time) from the total time of 4 hours, and use this to calculate s."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            input_infos_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
            if r > 0:
                input_infos_3.extend(all_thinking3[r-1])
            thinking3, answer3 = agent(input_infos_3, debate_instruction_3, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating walking time s, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make final decision on calculation of s.", is_sub_task=True)
    agents.append(f"Final Decision agent calculating s, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Subtask 4: Calculate the actual walking time for the second scenario using CoT.
    cot_instruction_4 = "Sub-task 4: Calculate the actual walking time in the second scenario, by subtracting t (coffee shop time) from the total time of 2 hours and 24 minutes, and confirm the value of s calculated in the previous subtask."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, confirming calculation of walking time, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Subtask 5: Compute Aya's total time for walking 9 kilometers at s + 1/2 km/h using Reflexion pattern.
    reflect_instruction_5 = "Sub-task 5: Compute Aya's total time for walking 9 kilometers at s + 1/2 kilometers per hour including the same t minutes at the coffee shop."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, calculating time for s + 1/2 km/h, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = critic_agent_5([taskInfo, thinking5, answer5], "Please review the calculation of the time at s + 1/2 km/h and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining time calculation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Subtask 6: Calculate the final duration in minutes using Reflexion.
    reflect_instruction_6 = "Sub-task 6: Derive the total duration in minutes that Aya takes for the walk at s + 1/2 kilometers per hour, combining the walking time with the coffee shop time t."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    thinking6, answer6 = cot_agent_6(cot_inputs_6, reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, deriving total duration, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = critic_agent_6([taskInfo, thinking6, answer6], "Please review the total duration calculation and provide feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining total duration walk, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer