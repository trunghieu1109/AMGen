async def forward_165(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and classify elements
    # Sub-task 1: Analyze the given Lagrangian and field content
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given Lagrangian and field content to identify the particle content, "
        "their gauge quantum numbers, and the role of each term in the Lagrangian, including the singlet fermions N_iR, "
        "scalar doublet S, and singlet scalar phi, as well as the vacuum expectation values (VEVs) x and v."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing Lagrangian and field content, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Classify physical states from scalar sector after spontaneous symmetry breaking
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, classify the physical states arising from the scalar sector after spontaneous symmetry breaking, "
        "focusing on the identification of the pseudo-Goldstone boson H_2 and its relation to the fields phi and h, given the VEVs <phi>=x and <h>=v."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying scalar physical states, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by majority vote
    from collections import Counter
    answer_counts_2 = Counter(possible_answers_2)
    best_answer_2 = answer_counts_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Extract radiative corrections and evaluate mass formula
    # Sub-task 3: Extract relevant radiative correction contributions to the mass of H_2
    cot_instruction_3 = (
        "Sub-task 3: Extract the relevant radiative correction contributions to the mass of the pseudo-Goldstone boson H_2 from the particle spectrum, "
        "including contributions from scalars (h_1, H^\u00B1, H^0, A^0), gauge bosons (W, Z), fermions (top quark t, singlet fermions N_i), and their respective coupling coefficients alpha_i."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, extracting radiative corrections, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate structure of one-loop effective potential or mass correction formula
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the structure of the one-loop effective potential or mass correction formula for the pseudo-Goldstone boson H_2, "
        "focusing on the dependence on the VEV combination (x^2 + v^2), the loop factor 1/(8pi^2), and the signs and powers of the mass terms in the formula."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating mass correction formula structure, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compare candidate mass formulas against derived theoretical structure using Debate
    debate_instruction_5 = (
        "Sub-task 5: Compare the candidate mass formulas (choices 1 to 4) for M_h_2^2 against the derived theoretical structure from previous subtasks, "
        "checking for correct factors, signs, included particle contributions, and overall consistency with the radiative correction approximation for the pseudo-Goldstone boson mass."
    )
    debate_roles = ["Proponent", "Opponent"]
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
            agents.append(f"Debate agent {agent.id}, round {r}, debating candidate formulas, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct approximation formula for the mass of the pseudo-Goldstone boson H_2.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding final formula, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
