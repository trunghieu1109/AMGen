async def forward_107(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Calculate transition energy ΔE and paramagnetic coupling term ⟨H⟩

    # Sub-task 1: Calculate transition energy ΔE of Hydrogen atom for λ = 0.4861 μm using E = hc/λ
    cot_instruction_1 = (
        "Sub-task 1: Calculate the transition energy ΔE of the Hydrogen atom corresponding to the given wavelength "
        "λ = 0.4861 μm using the photon energy formula E = hc/λ. Provide the energy in Joules and eV."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating transition energy ΔE, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine paramagnetic coupling term ⟨H⟩ = -μ_B m B for B=1T and small m
    cot_instruction_2 = (
        "Sub-task 2: Determine the paramagnetic coupling term ⟨H⟩ for the Hydrogen atom in a magnetic field B = 1 T, "
        "with small orbital magnetic quantum number m, using the formula ⟨H⟩ = -μ_B m B. Provide the energy in Joules and eV."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating paramagnetic coupling ⟨H⟩, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Convert energies into consistent units (Joules or eV) for comparison

    # Sub-task 3: Convert ΔE and ⟨H⟩ into consistent units and summarize values
    cot_sc_instruction_3 = (
        "Sub-task 3: Convert all calculated energies (ΔE and ⟨H⟩) into consistent units (Joules and eV) "
        "to enable direct numerical comparison of their magnitudes. Summarize the values clearly."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, converting energies to consistent units, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most frequent answer for consistency
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 1 continued: Evaluate order of magnitude comparison

    # Sub-task 4: Evaluate order of magnitude of ⟨H⟩ relative to ΔE
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the order of magnitude of the paramagnetic coupling term ⟨H⟩ relative to the transition energy ΔE "
        "by comparing their numerical values obtained in consistent units. Provide a clear comparison statement."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, comparing magnitudes of ⟨H⟩ and ΔE, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Select correct relationship choice based on comparison

    # Sub-task 5: Select correct relationship between ⟨H⟩ and ΔE from given choices
    debate_instruction_5 = (
        "Sub-task 5: Based on the comparison, select the correct relationship between ⟨H⟩ and ΔE from the given choices: "
        "⟨H⟩ ≪ ΔE, ⟨H⟩ ≫ ΔE, ⟨H⟩ = ΔE, or ⟨H⟩ > ΔE. Justify the choice clearly."
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
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct relationship, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct relationship between ⟨H⟩ and ΔE.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final relationship, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
