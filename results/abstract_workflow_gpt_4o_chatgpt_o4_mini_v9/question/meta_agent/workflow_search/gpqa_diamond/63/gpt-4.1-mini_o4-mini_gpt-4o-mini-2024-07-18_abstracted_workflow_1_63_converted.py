async def forward_63(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze problem statement and classify reagents
    # Sub-task 1: Analyze problem statement to identify chemical species, gases, reagents, and quantitative data
    cot_instruction_1 = (
        "Sub-task 1: Analyze the problem statement to identify the chemical species involved: salts A and B, "
        "the gases formed upon heating, and the reagents used in the gas analysis (anhydrous Mg(ClO4)2, Ca(OH)2 solution, red-hot copper). "
        "Extract all given quantitative data such as masses, volume of gas C, and conditions (temperature, no air)."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem statement, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Classify roles and chemical behavior of each reagent
    cot_instruction_2 = (
        "Sub-task 2: Classify the roles and chemical behavior of each reagent used in the gas analysis: "
        "determine what each tube absorbs or reacts with (e.g., Mg(ClO4)2 absorbs water, Ca(OH)2 absorbs acidic gases like CO2, red-hot copper reacts with oxygen to form CuO)."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, classifying reagent roles, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Derive gas mixture composition and identify salts
    # Sub-task 3: Derive gas mixture composition from weight changes and gas volume
    cot_instruction_3 = (
        "Sub-task 3: Derive the composition of the gas mixture formed by heating the equimolar mixture of salts A and B "
        "based on the weight changes in tubes №1 and №3 and the unchanged weight of tube №2, and the volume of gas C remaining after passing through all tubes."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, deriving gas mixture composition, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer (majority vote)
    from collections import Counter
    answer_counts_3 = Counter(possible_answers_3)
    best_answer_3 = answer_counts_3.most_common(1)[0][0]
    thinking3 = thinkingmapping_3[best_answer_3]
    answer3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Identify possible chemical formulas of salts A and B
    cot_instruction_4 = (
        "Sub-task 4: Identify the possible chemical formulas of salts A and B consistent with the gas composition and the stoichiometry implied by the problem "
        "(equimolar mixture, total mass, and gas volume)."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, identifying chemical formulas of salts, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Calculate total number of atoms in combined formula units
    # Sub-task 5: Calculate total number of atoms using derived formulas and stoichiometry
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, calculate the total number of atoms in the combined formula units of salts A and B "
        "using the derived chemical formulas and the stoichiometric relationships from the previous subtasks."
    )
    debate_roles = ["Agent 1", "Agent 2"]
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
            agents.append(f"Debate agent {agent.id}, round {r}, calculating total atoms, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on total number of atoms.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating total atoms, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Stage 3: Compare with answer choices and select correct answer
    cot_instruction_6 = (
        "Sub-task 6: Compare the calculated total number of atoms with the provided answer choices (15, 13, 17, 19) and select the correct answer."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting correct answer choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 3.6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
