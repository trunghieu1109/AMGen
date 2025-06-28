async def forward_51(self, taskInfo):
    from collections import Counter
    import math

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 0: Analyze and extract physical parameters and assumptions
    cot_instruction_0_1 = (
        "Sub-task 1: Extract and analyze all given physical parameters of the star and observational conditions: "
        "radius (1.5 solar radii), mass (1.1 solar masses), effective temperature without spots (6000 K), "
        "effective temperature with 40% spot coverage (5500 K), spot coverage fraction (40%), and the wavelength of the Ti transition (1448 Å). "
        "Identify that the photosphere is in LTE and that the ratio of neutral Ti atoms in two energy levels changes with spot coverage."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_1, answer0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracted physical parameters, thinking: {thinking0_1.content}; answer: {answer0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Confirm and classify the physical context and assumptions: verify that the stellar photosphere is in Local Thermodynamic Equilibrium (LTE), "
        "allowing the use of the Boltzmann distribution to relate population ratios of atomic energy levels to temperature. "
        "Recognize that the observed change in the ratio of neutral Ti atoms in two energy levels is due to the change in effective temperature caused by spot coverage."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_2, answer0_2 = await cot_agent_0_2([taskInfo, thinking0_1, answer0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, classified physical context, thinking: {thinking0_2.content}; answer: {answer0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    cot_instruction_0_3 = (
        "Sub-task 3: Determine the effective temperature of the spotted regions given the composite effective temperature (5500 K), "
        "spot coverage fraction (40%), and the known temperature of the unspotted regions (6000 K). "
        "This helps understand the temperature of the spotted regions if needed."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_3, answer0_3 = await cot_agent_0_3([taskInfo, thinking0_1, answer0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, calculated spotted region temperature, thinking: {thinking0_3.content}; answer: {answer0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking0_3.content}; answer - {answer0_3.content}")
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Calculate energy difference and apply Boltzmann distribution
    cot_instruction_1_4 = (
        "Sub-task 4: Calculate the energy difference (ΔE) between the two Ti energy levels using the given wavelength (1448 Å) of the transition, "
        "applying ΔE = hc/λ, where h is Planck's constant and c is the speed of light. "
        "This energy difference is needed to apply the Boltzmann equation for level populations."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1_4, answer1_4 = await cot_agent_1_4([taskInfo, thinking0_1, answer0_1], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, calculated energy difference ΔE, thinking: {thinking1_4.content}; answer: {answer1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking1_4.content}; answer - {answer1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    cot_instruction_1_5 = (
        "Sub-task 5: Apply the Boltzmann distribution formula to express the ratio of the number of neutral Ti atoms in the two energy levels as a function of temperature and energy difference ΔE. "
        "This formula relates the population ratio to temperature, which will be used to compare the ratios at different effective temperatures."
    )
    cot_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1_5, answer1_5 = await cot_agent_1_5([taskInfo, thinking0_2, answer0_2, thinking1_4, answer1_4], cot_instruction_1_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_5.id}, formulated Boltzmann ratio expression, thinking: {thinking1_5.content}; answer: {answer1_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking1_5.content}; answer - {answer1_5.content}")
    print("Step 1.5: ", sub_tasks[-1])

    cot_instruction_1_6 = (
        "Sub-task 6: Calculate the ratio of the number of neutral Ti atoms in the two energy levels at the effective temperature without spots (6000 K) using the Boltzmann distribution and the energy difference ΔE."
    )
    cot_agent_1_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1_6, answer1_6 = await cot_agent_1_6([taskInfo, thinking1_5, answer1_5], cot_instruction_1_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_6.id}, calculated ratio at 6000K, thinking: {thinking1_6.content}; answer: {answer1_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking1_6.content}; answer - {answer1_6.content}")
    print("Step 1.6: ", sub_tasks[-1])

    cot_instruction_1_7 = (
        "Sub-task 7: Calculate the ratio of the number of neutral Ti atoms in the two energy levels at the effective temperature with spots (5500 K) using the Boltzmann distribution and the energy difference ΔE."
    )
    cot_agent_1_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1_7, answer1_7 = await cot_agent_1_7([taskInfo, thinking1_5, answer1_5], cot_instruction_1_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_7.id}, calculated ratio at 5500K, thinking: {thinking1_7.content}; answer: {answer1_7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking1_7.content}; answer - {answer1_7.content}")
    print("Step 1.7: ", sub_tasks[-1])

    # Stage 2: Compute ratio change factor and identify closest choice
    cot_instruction_2_8 = (
        "Sub-task 8: Compute the factor by which the ratio of the number of neutral Ti atoms in the two energy levels changes when the star does not have spots compared to when it has spots, "
        "by dividing the ratio at 6000 K by the ratio at 5500 K."
    )
    cot_agent_2_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2_8, answer2_8 = await cot_agent_2_8([taskInfo, thinking1_6, answer1_6, thinking1_7, answer1_7], cot_instruction_2_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_8.id}, computed ratio change factor, thinking: {thinking2_8.content}; answer: {answer2_8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking2_8.content}; answer - {answer2_8.content}")
    print("Step 2.8: ", sub_tasks[-1])

    debate_instruction_2_9 = (
        "Sub-task 9: Compare the computed factor with the provided multiple-choice options (~2.9, ~4.5, ~7.8, ~1.1) and identify the closest matching value as the final answer."
    )
    debate_agents_2_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_9 = self.max_round
    all_thinking2_9 = [[] for _ in range(N_max_2_9)]
    all_answer2_9 = [[] for _ in range(N_max_2_9)]

    for r in range(N_max_2_9):
        for i, agent in enumerate(debate_agents_2_9):
            input_infos_2_9 = [taskInfo, thinking2_8, answer2_8]
            if r > 0:
                input_infos_2_9 += all_thinking2_9[r-1] + all_answer2_9[r-1]
            thinking2_9, answer2_9 = await agent(input_infos_2_9, debate_instruction_2_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing factor to choices, thinking: {thinking2_9.content}; answer: {answer2_9.content}")
            all_thinking2_9[r].append(thinking2_9)
            all_answer2_9[r].append(answer2_9)

    final_decision_agent_2_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_9, answer2_9 = await final_decision_agent_2_9([taskInfo] + all_thinking2_9[-1] + all_answer2_9[-1], "Sub-task 9: Make final decision on the closest matching factor to the computed ratio change.", is_sub_task=True)
    agents.append(f"Final Decision agent on closest factor, thinking: {thinking2_9.content}; answer: {answer2_9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking2_9.content}; answer - {answer2_9.content}")
    print("Step 2.9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_9, answer2_9, sub_tasks, agents)
    return final_answer
