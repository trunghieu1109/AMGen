async def forward_91(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Compute quantitative measures

    # Sub-task 1: Identify and count the number of each type of atom (C, H) in the molecule
    cot_instruction_1 = (
        "Sub-task 1: Identify and count the number of carbon (C) and hydrogen (H) atoms in the molecule "
        "(CH3)2C=CH-CH2-CH(CH3)-CH2-CH=C(CH3)2 to determine total atoms for enthalpy calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, counting atoms, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine number and types of bonds (C-C single, C=C double, C-H) present
    cot_instruction_2 = (
        "Sub-task 2: Based on atom counts from Sub-task 1, determine the number and types of bonds "
        "(C-C single, C=C double, C-H) in the molecule to calculate total bond energies."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, determining bonds, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate total enthalpy of atomization for all carbon atoms
    cot_instruction_3 = (
        "Sub-task 3: Calculate total enthalpy of atomization for all carbon atoms using the given enthalpy "
        "of atomization per carbon atom and total carbon count from Sub-task 1."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating enthalpy of atomization, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Calculate total bond energy for all bonds (C-C, C=C, C-H)
    cot_instruction_4 = (
        "Sub-task 4: Calculate total bond energy for all bonds (C-C single, C=C double, C-H) in the molecule "
        "using bond counts from Sub-task 2 and given bond energies."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating total bond energy, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Combine and finalize

    # Sub-task 5: Compute enthalpy of formation by combining total enthalpy of atomization and total bond energies
    debate_instruction_5 = (
        "Sub-task 5: Compute the enthalpy of formation of the molecule by combining total enthalpy of atomization "
        "of carbon (Sub-task 3) and total bond energies (Sub-task 4) according to thermodynamic relations."
    )
    debate_roles = ["Thermodynamics Expert", "Physical Chemist"]
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True
                )
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing enthalpy of formation, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Make final decision on the enthalpy of formation value.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, deciding enthalpy of formation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare calculated enthalpy of formation with given choices to identify correct answer
    cot_instruction_6 = (
        "Sub-task 6: Compare the calculated enthalpy of formation value from Sub-task 5 with the given choices "
        "(11200 kJ/mol, 1900 kJ/g, 11.44 kJ/g, 67.4 kJ/mol) to identify the correct answer."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
