async def forward_177(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Identify mass dimensions of fields and operators
    cot_instruction_1 = (
        "Sub-task 1: Identify and write down the mass dimensions of the fundamental fields and operators "
        "appearing in the interaction Lagrangian L_int = kappa * \bar{\psi} * sigma_{\mu\nu} * \psi * F^{\mu\nu}, specifically the mass dimensions of the fermion field \psi, "
        "the field strength tensor F^{\mu\nu}, and the gamma matrix commutator sigma_{\mu\nu}, using known QFT conventions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying mass dimensions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1: Combine mass dimensions to find total operator dimension
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the mass dimensions identified in Sub-task 1, combine the mass dimensions of the fields and operators "
        "to determine the total mass dimension of the operator \bar{\psi} sigma_{\mu\nu} \psi F^{\mu\nu} appearing in the interaction term, excluding the coupling constant kappa. "
        "Consider all possible cases and ensure consistency with QFT conventions."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, combining mass dimensions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by majority vote
    counter_2 = Counter(possible_answers_2)
    best_answer_2 = counter_2.most_common(1)[0][0]
    best_thinking_2 = thinkingmapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {best_thinking_2.content}; answer - {best_answer_2}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 1: Derive mass dimension of coupling constant kappa
    cot_instruction_3 = (
        "Sub-task 3: Using the fact that the Lagrangian density must have mass dimension 4 in 4-dimensional spacetime, "
        "derive the mass dimension of the coupling constant kappa by balancing the dimensions in the interaction term. "
        "Use the total operator dimension found in Sub-task 2."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, best_thinking_2, answermapping_2[best_answer_2]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving mass dimension of kappa, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Review renormalizability criteria
    cot_instruction_4 = (
        "Sub-task 4: Review the criteria for renormalizability of a quantum field theory interaction term based on the mass dimension of the coupling constant kappa, "
        "specifically that couplings with non-negative mass dimension correspond to renormalizable interactions."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, reviewing renormalizability criteria, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Determine if theory is renormalizable
    debate_instruction_5 = (
        "Sub-task 5: Based on the mass dimension of kappa found in Sub-task 3 and the renormalizability criteria reviewed in Sub-task 4, "
        "determine whether the theory described by the given interaction Lagrangian is renormalizable or not. "
        "Debate agents should consider the implications of the mass dimension and criteria to reach a conclusion."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining renormalizability, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on whether the theory is renormalizable based on previous outputs.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining renormalizability, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
