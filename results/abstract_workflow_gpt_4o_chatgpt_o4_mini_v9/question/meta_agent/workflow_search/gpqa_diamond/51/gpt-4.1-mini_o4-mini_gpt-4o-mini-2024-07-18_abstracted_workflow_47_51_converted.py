async def forward_51(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Extract parameters and understand physical context
    # Sub-task 1: Identify and extract all given physical parameters
    cot_instruction_1 = (
        "Sub-task 1: Identify and extract all given physical parameters of the star relevant to the problem: "
        "radius (1.5 solar radii), mass (1.1 solar masses), effective temperature without spots (6000 K), "
        "effective temperature with 40% spot coverage (5500 K), spot coverage fraction (40%), and the wavelength corresponding to the energy level transition (1448 Å). "
        "This sets the foundation for further calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting physical parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Understand and characterize the physical context
    cot_instruction_2 = (
        "Sub-task 2: Understand and characterize the physical context: the star's photosphere is in Local Thermodynamic Equilibrium (LTE), "
        "and the ratio of neutral Ti atoms in two energy levels changes with temperature. Recognize that the observed ratio corresponds to a Boltzmann distribution dependent on temperature and energy difference between levels."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, characterizing physical context, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Calculate energy difference and formulate Boltzmann ratio
    # Sub-task 3: Calculate energy difference ΔE = hc/λ
    cot_instruction_3 = (
        "Sub-task 3: Calculate the energy difference (ΔE) between the two Ti energy levels using the given wavelength (1448 Å) of the transition, "
        "applying the relation ΔE = hc/λ, where h is Planck's constant and c is the speed of light. "
        "This energy difference is needed to evaluate the Boltzmann factor for the level populations."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating energy difference ΔE, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Formulate Boltzmann ratio expression
    cot_instruction_4 = (
        "Sub-task 4: Formulate the expression for the ratio of the number of neutral Ti atoms in the two energy levels under LTE conditions using the Boltzmann distribution: "
        "N2/N1 = (g2/g1) * exp(-ΔE / (k_B * T)), where g1 and g2 are statistical weights (assumed constant or cancel out), "
        "ΔE is from subtask 3, k_B is Boltzmann constant, and T is the effective temperature of the photosphere."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, formulating Boltzmann ratio, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Calculate or confirm effective temperature with spots
    cot_instruction_5 = (
        "Sub-task 5: Calculate the effective temperature of the star's photosphere when 40% of the surface is covered by spots, "
        "given the overall effective temperature is 5500 K and the spot coverage fraction is 40%, with the spot temperature assumed to be lower than the unspotted photosphere temperature (6000 K). "
        "This may involve solving for the spot temperature or confirming the given effective temperature is the combined effect."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating effective temperature with spots, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compute N2/N1 ratios for no spots and with spots
    cot_instruction_6 = (
        "Sub-task 6: Compute the ratio of the number of neutral Ti atoms in the two energy levels (N2/N1) for the star without spots (T = 6000 K) "
        "and with spots (T = 5500 K) using the Boltzmann expression from subtask 4 and the temperatures identified."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, computing N2/N1 ratios, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Determine factor change of N2/N1 ratio no spots vs with spots
    debate_instruction_7 = (
        "Sub-task 7: Determine the factor by which the ratio N2/N1 changes when the star does not have spots compared to when it has spots by calculating "
        "(N2/N1)_no_spots / (N2/N1)_with_spots. This final step answers the original question and allows comparison with the provided multiple-choice options."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            input_infos_7 = [taskInfo, thinking6, answer6]
            if r > 0:
                input_infos_7.extend(all_thinking7[r-1])
            thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining factor change, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the factor by which the ratio N2/N1 changes.", is_sub_task=True)
    agents.append(f"Final Decision agent on factor change, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
