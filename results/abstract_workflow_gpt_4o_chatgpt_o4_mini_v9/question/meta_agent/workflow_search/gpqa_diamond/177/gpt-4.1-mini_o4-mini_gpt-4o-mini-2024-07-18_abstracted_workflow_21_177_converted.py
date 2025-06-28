async def forward_177(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze fields and operators involved
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given Lagrangian interaction term \( \mathcalL_{int} = \kappa \bar{\psi} \sigma_{\mu\nu} \psi F^{\mu\nu} \) and identify the fields and operators involved, "
        "including their known mass dimensions and physical meaning. Recognize \( \psi \) as a fermion field, \( F^{\mu\nu} \) as the field strength tensor, and \( \sigma_{\mu\nu} = \frac{i}{2}[\gamma_{\mu}, \gamma_{\nu}] \) as a spin matrix constructed from gamma matrices."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing fields and operators, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 0: Determine canonical mass dimensions of each component
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the identification from Sub-task 1, determine the canonical mass dimensions of each component in the interaction term: "
        "the fermion field \( \psi \), the field strength tensor \( F^{\mu\nu} \), and the spin matrix \( \sigma_{\mu\nu} \), using standard quantum field theory conventions in 4-dimensional spacetime."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining mass dimensions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping_2[most_common_answer_2].content}; answer - {most_common_answer_2}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Compute overall mass dimension of the operator
    cot_instruction_3 = (
        "Sub-task 3: Using the mass dimensions of the fields and operators identified in Sub-task 2, compute the overall mass dimension of the operator "
        "\( \bar{\psi} \sigma_{\mu\nu} \psi F^{\mu\nu} \) in the interaction Lagrangian."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinkingmapping_2[most_common_answer_2], answermapping_2[most_common_answer_2]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, computing overall mass dimension, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Calculate mass dimension of coupling constant kappa
    cot_instruction_4 = (
        "Sub-task 4: From the total mass dimension of the interaction term and the fact that the Lagrangian density must have mass dimension 4 in 4-dimensional spacetime, "
        "calculate the mass dimension of the coupling constant \( \kappa \)."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating mass dimension of coupling constant, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Evaluate renormalizability based on mass dimension of kappa
    cot_instruction_5 = (
        "Sub-task 5: Evaluate the renormalizability of the theory based on the mass dimension of the coupling constant \( \kappa \) found in Sub-task 4. "
        "Use the criterion that couplings with non-negative mass dimension correspond to renormalizable interactions, while negative mass dimension couplings indicate non-renormalizable interactions."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, evaluating renormalizability, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Compare with multiple-choice options
    debate_instruction_6 = (
        "Sub-task 6: Compare the computed mass dimension and renormalizability conclusion with the provided multiple-choice options to identify which choice correctly matches the results."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            input_infos_6 = [taskInfo, thinking4, answer4, thinking5, answer5]
            if r > 0:
                input_infos_6 += all_thinking6[r-1] + all_answer6[r-1]
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent on multiple-choice answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
