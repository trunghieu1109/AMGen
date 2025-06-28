async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning
    # Sub-task 1: Understand relationships between speed, walk time, and coffee shop time.
    cot_instruction_1 = "Sub-task 1: Understand the relationship between Aya's walking speed, time spent walking, and time spent at the coffee shop based on the given scenarios."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding relationship, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Calculate Ayas walking speed and coffee shop time.
    cot_sc_instruction_2 = "Sub-task 2: Calculate Aya's walking speed (s) and time spent at the coffee shop based on her speeds of s and s+2 km/h."
    N_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N_2):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating walking speed and time, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Determine walk time when walking at s+2 km/h.
    cot_reflect_instruction_3 = "Sub-task 3: Determine the walk time for Aya when she walks at s+2 km/h, given that the total time is 2 hours and 24 minutes with the coffee shop time."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining walk time with s+2 km/h, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback_3, correct_3 = critic_agent_3([taskInfo, thinking3, answer3], "please review the walk time with s+2 km/h and provide feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback_3.content}; answer: {correct_3.content}")
        if correct_3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback_3])
        thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining walk time determination, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Intermediate Inference and Validation
    # Sub-task 4: Validate calculated speeds and times.
    cot_reflect_instruction_4 = "Sub-task 4: Validate the calculated walking speeds and coffee shop time based on given constraints."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round

    cot_inputs_4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, validating speeds and times, thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max_4):
        feedback_4, correct_4 = critic_agent_4([taskInfo, thinking4, answer4], "please review the speed and time validation and provide feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback_4.content}; answer: {correct_4.content}")
        if correct_4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback_4])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining validation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 3: Final Output Generation and Integration
    # Sub-task 5: Calculate total time when Aya walks at s+0.5 km/h.
    debate_instruction_5 = "Sub-task 5: Using validated speeds and times, calculate total time for Aya walking at s+0.5 km/h including coffee shop time."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round

    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for agent in debate_agents_5:
            input_infos_5 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos_5.extend(all_thinking5[r-1])
            thinking5, answer5 = agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating final time, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_answers_5 = [ans.content for ans in all_answer5[-1]]
    final_answercontent_5 = Counter(final_answers_5).most_common(1)[0][0]
    index_5 = final_answers_5.index(final_answercontent_5)
    thinking5 = all_thinking5[-1][index_5]
    answer5 = all_answer5[-1][index_5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer