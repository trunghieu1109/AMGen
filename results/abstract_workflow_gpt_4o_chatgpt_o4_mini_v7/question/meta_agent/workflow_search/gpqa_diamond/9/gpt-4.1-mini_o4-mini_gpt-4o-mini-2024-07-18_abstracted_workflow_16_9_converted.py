async def forward_9(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Standardize planet data
    cot_instruction_0 = (
        "Sub-task 1: Standardize the given planet data by extracting and defining key parameters for each planet: "
        "mass (in Earth masses), radius (in Earth radii), density (in g/cm^3), and composition assumptions. "
        "Prepare the data for consistent comparison across all planets."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, standardizing planet data, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Calculate densities for each planet
    # Sub-task 2: Calculate density for planet a (Earth-mass and Earth-radius)
    cot_instruction_2 = (
        "Sub-task 2: Calculate the density of the planet described as an Earth-mass and Earth-radius planet (choice a) "
        "using the standard density formula, given mass and radius equal to Earth's values."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking0, answer0], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating density for planet a, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify density for planet b (2 Earth masses, density ~5.5 g/cm^3)
    cot_instruction_3 = (
        "Sub-task 3: Identify and record the density of the planet with 2 Earth masses and a given density of approximately 5.5 g/cm^3 (choice b) "
        "directly from the provided data."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking0, answer0], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, identifying density for planet b, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Calculate density for planet c (same composition as Earth, 5x Earth mass)
    cot_instruction_4 = (
        "Sub-task 4: Calculate the density of the planet with the same composition as Earth but 5 times more massive (choice c). "
        "Estimate the radius based on mass-radius relationship for Earth-like composition, then compute density using mass and radius."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking0, answer0], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating density for planet c, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Calculate density for planet d (same composition as Earth, half Earth mass)
    cot_instruction_5 = (
        "Sub-task 5: Calculate the density of the planet with the same composition as Earth but half the mass (choice d). "
        "Estimate the radius based on mass-radius relationship for Earth-like composition, then compute density using mass and radius."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking0, answer0], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating density for planet d, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Compare densities and determine highest
    # Sub-task 6: Compare densities of all four planets
    cot_instruction_6 = (
        "Sub-task 6: Compare the densities calculated or given for all four planets (choices a, b, c, d) "
        "to determine which planet has the highest density."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6(
        [taskInfo, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5],
        cot_instruction_6, is_sub_task=True
    )
    agents.append(f"CoT agent {cot_agent_6.id}, comparing densities, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Provide final answer identifying planet with highest density
    debate_instruction_7 = (
        "Sub-task 7: Based on the comparison of densities, provide a final answer identifying the planet with the highest density among the given options, "
        "supported by the calculations and comparisons performed."
    )
    debate_agents_7 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            input_infos_7 = [taskInfo, thinking6, answer6]
            if r > 0:
                input_infos_7.extend(all_thinking7[r-1])
                input_infos_7.extend(all_answer7[r-1])
            thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, providing final answer, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7(
        [taskInfo] + all_thinking7[-1] + all_answer7[-1],
        "Sub-task 7: Make final decision on the planet with the highest density.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, finalizing answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
