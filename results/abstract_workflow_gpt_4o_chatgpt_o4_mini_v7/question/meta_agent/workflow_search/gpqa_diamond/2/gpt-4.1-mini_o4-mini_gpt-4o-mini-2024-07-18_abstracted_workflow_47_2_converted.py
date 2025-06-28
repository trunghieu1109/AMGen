async def forward_2(self, taskInfo):
    from collections import Counter
    import math

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 1: Identify quantum state and operators

    # Sub-task 1: Identify and write down the given quantum state vector and confirm normalization
    cot_instruction_1 = (
        "Sub-task 1: Identify the given quantum state vector |\u2191> and |\u2193> basis with coefficients 0.5 and sqrt(3)/2, "
        "and confirm its normalization."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying quantum state, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Express sigma_z and sigma_x matrices in |\u2191>, |\u2193> basis
    cot_instruction_2 = (
        "Sub-task 2: Express the operators sigma_z and sigma_x in matrix form in the |\u2191>, |\u2193> basis, "
        "noting that |\u2191> and |\u2193> are eigenstates of sigma_z."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, expressing sigma matrices, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Construct combined operator 10*sigma_z + 5*sigma_x
    cot_instruction_3 = (
        "Sub-task 3: Construct the combined operator 10*sigma_z + 5*sigma_x as a matrix by linearly combining the matrices "
        "of sigma_z and sigma_x with coefficients 10 and 5 respectively."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, constructing combined operator, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Calculate expectation value and evaluate numerical result

    # Sub-task 4: Calculate expectation value <psi| (10*sigma_z + 5*sigma_x) |psi>
    cot_instruction_4 = (
        "Sub-task 4: Calculate the expectation value of the operator 10*sigma_z + 5*sigma_x in the given quantum state |psi> "
        "by performing the matrix multiplication <psi| (10*sigma_z + 5*sigma_x) |psi>."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating expectation value, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate numerical value and round to one decimal place
    cot_instruction_5 = (
        "Sub-task 5: Evaluate the numerical value of the expectation value obtained in Sub-task 4, simplify it to a decimal value, "
        "and round it to one decimal place as required."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, evaluating numerical value, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare rounded expectation value with multiple-choice options
    debate_instruction_6 = (
        "Sub-task 6: Compare the rounded expectation value from Sub-task 5 with the provided multiple-choice options: 0.85, 1.65, -1.4, -0.7, "
        "and identify the closest matching choice."
    )
    debate_agents_6 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            input_infos_6 = [taskInfo, thinking5, answer5]
            if r > 0:
                input_infos_6.extend(all_thinking6[r-1])
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6(
        [taskInfo] + all_thinking6[-1] + all_answer6[-1],
        "Sub-task 6: Make final decision on the closest matching multiple-choice answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent on matching choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
