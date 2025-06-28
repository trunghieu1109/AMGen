async def forward_108(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Extract and clarify quantum numbers and conditions
    cot_instruction_1 = (
        "Sub-task 1: Extract and clarify the quantum numbers and properties of the initial NN bound state (1S0), "
        "including total spin (S), orbital angular momentum (L), total angular momentum (J), and isospin (T), "
        "as well as the intrinsic parity of the emitted particle X (-1). This sets the foundational parameters for the problem."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting initial quantum numbers, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Interpret Pauli statistics condition and final isospin
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, interpret and restate the Pauli statistics condition "
        "T(NN) = S(NN) + L(NN) + 1 (mod 2) for the final NN state, and understand the implication of the given final isospin T(NN) = 0 "
        "on the allowed combinations of spin S(NN) and orbital angular momentum L(NN)."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, interpreting Pauli condition, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote)
    from collections import Counter
    answer_counts_2 = Counter(possible_answers_2)
    best_answer_2 = answer_counts_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Analyze conservation laws of angular momentum and parity
    cot_instruction_3 = (
        "Sub-task 3: Analyze the conservation laws relevant to the transition: total angular momentum conservation "
        "(initial J = final J + angular momentum of emitted particle X) and parity conservation, taking into account the intrinsic parity of X and the orbital angular momentum of X (denoted by lowercase letters s, p, d, f, etc.)."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing conservation laws, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: For each choice, verify Pauli condition and conservation laws
    cot_reflect_instruction_4 = (
        "Sub-task 4: For each given choice (1S0 -> 3P0 + s, 1S0 -> 7D1 + p, 1S0 -> 3D3 + f, 1S0 -> 3S1 + p), extract the quantum numbers "
        "of the final NN state (S, L, J) and the angular momentum state of the emitted particle X, and verify if the Pauli statistics condition "
        "T(NN) = S(NN) + L(NN) + 1 (mod 2) holds with T(NN) = 0."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_reflect_instruction_4, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, verifying Pauli condition for choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    cot_reflect_instruction_5 = (
        "Sub-task 5: For each choice, check the conservation of total angular momentum and parity between the initial state and the final NN state plus emitted particle X, "
        "considering the intrinsic parity of X and the orbital angular momentum of X."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3, thinking4, answer4], cot_reflect_instruction_5, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, checking conservation laws for choices, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Identify which partial waves violate conditions
    debate_instruction_6 = (
        "Sub-task 6: Identify which of the given partial waves (choices) violate either the Pauli statistics condition or the conservation laws of angular momentum and parity, "
        "and thus are not permitted final states."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, identifying invalid partial waves, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on which partial waves are not permitted.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining not permitted partial waves, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
