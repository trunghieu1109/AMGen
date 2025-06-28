async def forward_14(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Identify and analyze given parameters
    
    # Sub-task 1: Identify and list all given parameters and known information about Planet_1 and Planet_2
    cot_instruction_1 = (
        "Sub-task 1: Identify and list all given parameters and known information about Planet_1 and Planet_2, "
        "including their masses, orbital periods, orbit shapes, and host star properties (mass and radius). "
        "This sets the foundation for further calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, listing known parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze the relationship between the orbital periods and confirm circular orbits
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, analyze the relationship between the orbital periods of Planet_1 and Planet_2, "
        "noting that Planet_1's period is three times shorter than Planet_2's, and confirm both have circular orbits. "
        "This will be used to relate orbital radius and transit probability."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing orbital periods and orbits, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze host star properties
    cot_instruction_3 = (
        "Sub-task 3: Analyze the host star properties: confirm that the host star of Planet_1 has twice the mass of the host star of Planet_2, "
        "but both stars have the same radius due to the slight evolution of Planet_2's host star. "
        "This will be important for calculating orbital radii and transit probabilities."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing host star properties, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Calculate semi-major axes and transit probabilities

    # Sub-task 4: Calculate semi-major axes using Kepler's third law
    cot_reflect_instruction_4 = (
        "Sub-task 4: Calculate the semi-major axes (orbital radii) of Planet_1 and Planet_2 using Kepler's third law, "
        "incorporating the known orbital periods and host star masses from previous subtasks. "
        "This is essential to determine the geometric transit probability."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, calculating semi-major axes, thinking: {thinking4.content}; answer: {answer4.content}")
    # Critic agent for refinement
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_refine_4 = cot_inputs_4[:]
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4],
                                                  "Critically evaluate the semi-major axes calculation for correctness and completeness.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback on semi-major axes, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_refine_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_refine_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining semi-major axes, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Calculate geometric transit probabilities
    cot_reflect_instruction_5 = (
        "Sub-task 5: Calculate the geometric transit probability for each planet using the formula: transit probability ≈ (R_star / a), "
        "where R_star is the host star radius and a is the semi-major axis calculated in Sub-task 4. Use the fact that both stars have the same radius."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, calculating transit probabilities, thinking: {thinking5.content}; answer: {answer5.content}")
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_refine_5 = cot_inputs_5[:]
    for i in range(N_max_5):
        feedback5, correct5 = await critic_agent_5([taskInfo, thinking5, answer5],
                                                  "Critically evaluate the transit probability calculations for accuracy and completeness.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback on transit probabilities, thinking: {feedback5.content}; answer: {correct5.content}")
        if correct5.content == "True":
            break
        cot_inputs_refine_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent_5(cot_inputs_refine_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining transit probabilities, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Compare transit probabilities and map to choices

    # Sub-task 6: Compare transit probabilities and compute ratio
    cot_instruction_6 = (
        "Sub-task 6: Compare the transit probabilities of Planet_1 and Planet_2 calculated in Sub-task 5, "
        "compute the ratio of their transit probabilities, and determine which planet has the higher probability of transiting."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing transit probabilities, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Map computed ratio and preferred planet to closest matching choice
    debate_instruction_7 = (
        "Sub-task 7: Map the computed ratio and preferred planet from Sub-task 6 to the closest matching choice among the provided options (choice1 to choice4), "
        "to identify the researchers' preferred planet to observe."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            input_infos_7 = [taskInfo, thinking6, answer6]
            if r > 0:
                input_infos_7 += all_thinking7[r-1] + all_answer7[r-1]
            thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mapping ratio to choice, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1],
                                                    "Sub-task 7: Make final decision on the preferred planet to observe based on transit probability ratio.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent on preferred planet, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
