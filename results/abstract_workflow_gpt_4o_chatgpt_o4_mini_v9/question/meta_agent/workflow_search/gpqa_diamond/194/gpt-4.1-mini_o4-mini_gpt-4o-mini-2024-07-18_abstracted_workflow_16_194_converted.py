async def forward_194(self, taskInfo):
    from collections import Counter
    import math

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 1: Extract and standardize parameters (CoT with self-consistency)
    cot_instruction_1 = (
        "Sub-task 1: Extract and standardize all given parameters of the system: radius of the first planet (1 Earth radius), "
        "radius of the star (1.5 Solar radii), orbital period of the first planet (3 days), transit impact parameter (0.2), "
        "and radius of the second planet (2.5 Earth radii). Prepare these for further calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting and standardizing parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Calculate orbital radius of first planet (SC-CoT)
    cot_sc_instruction_2 = (
        "Sub-task 2: Using the standardized parameters from Sub-task 1, calculate the orbital radius (semi-major axis) "
        "of the first planet using its orbital period and star properties, assuming a circular orbit."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating orbital radius of first planet, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer for consistency
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Determine geometric constraints for transit and occultation (Reflexion)
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on parameters from Sub-task 1, determine the geometric constraints for a planet to exhibit both transit and occultation events. "
        "Calculate the maximum allowed impact parameter for both transit and occultation to occur, considering star radius, planet radius, and orbital inclination derived from the first planet's impact parameter."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining geometric constraints, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Review the geometric constraints calculation for transit and occultation and provide limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback on geometric constraints, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining geometric constraints, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Calculate max orbital radius for second planet (CoT)
    cot_instruction_4 = (
        "Sub-task 4: Using the geometric constraints from Sub-task 3, calculate the maximum orbital radius (semi-major axis) "
        "for the second planet that allows both transit and occultation, assuming same orbital plane and circular orbit."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating max orbital radius for second planet, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Convert max orbital radius to max orbital period (Debate)
    debate_instruction_5 = (
        "Sub-task 5: Convert the maximum orbital radius from Sub-task 4 into the corresponding maximum orbital period for the second planet "
        "using Kepler's third law and the star's mass estimated from its radius."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, converting orbital radius to period, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on maximum orbital period.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding max orbital period, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Compare max orbital period with choices and select closest (CoT)
    cot_instruction_6 = (
        "Sub-task 6: Compare the calculated maximum orbital period from Sub-task 5 with the provided multiple-choice options (~7.5, ~33.5, ~37.5, ~12.5 days) "
        "and select the closest value as the final answer."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting closest orbital period choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
