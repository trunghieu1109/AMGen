async def forward_91(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze molecular formula and structure
    cot_instruction_1 = (
        "Sub-task 1: Analyze the molecular formula and structure of (CH3)2C=CH-CH2-CH(CH3)-CH2-CH=C(CH3)2 to identify and count the number of carbon (C) and hydrogen (H) atoms, "
        "and classify the types and numbers of bonds (C-C single bonds, C=C double bonds, and C-H bonds) present in the molecule. "
        "Provide detailed counts for each atom and bond type."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1: Calculate total bond energy using bond counts
    cot_sc_instruction_2 = (
        "Sub-task 2: Using the bond counts from Sub-task 1, calculate the total bond energy of the molecule by summing the energies of all C-C, C=C, and C-H bonds "
        "using the given bond energies (C-C = 200 kJ/mol, C=C = 300 kJ/mol, C-H = 400 kJ/mol). Provide detailed calculation steps and final total bond energy."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating total bond energy, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote)
    from collections import Counter
    answer_counts_2 = Counter(possible_answers_2)
    most_common_answer_2 = answer_counts_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Calculate total enthalpy of atomization
    cot_instruction_3 = (
        "Sub-task 3: Calculate the total enthalpy of atomization of the molecule by multiplying the number of carbon atoms by the given enthalpy of atomization of carbon (1000 kJ/mol) "
        "and adding the enthalpy contribution from hydrogen atoms (assume 0 kJ/mol for hydrogen as no data given). Provide detailed calculation and final value."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating enthalpy of atomization, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Calculate enthalpy of formation
    cot_instruction_4 = (
        "Sub-task 4: Calculate the enthalpy of formation of the molecule by using the formula: Enthalpy of formation = Total bond energy (from Sub-task 2) - Total enthalpy of atomization (from Sub-task 3). "
        "Provide detailed calculation and final enthalpy of formation value."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating enthalpy of formation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Compare calculated enthalpy of formation with given choices using Debate
    debate_instruction_5 = (
        "Sub-task 5: Compare the calculated enthalpy of formation (from Sub-task 4) with the given answer choices (11200 kJ/mol, 1900 kJ/g, 11.44 kJ/g, 67.4 kJ/mol) "
        "to identify the correct or closest matching value, considering units and magnitude. Provide reasoning for the choice."
    )
    debate_roles = ["Proponent", "Skeptic"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
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
            agents.append(f"Debate agent {agent.id}, round {r}, debating correct choice, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct enthalpy of formation value.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
