async def forward_154(self, taskInfo):
    from collections import Counter
    import math
    import cmath

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 1: Extract and define operators and basis vectors
    cot_instruction_1 = (
        "Sub-task 1: Extract and clearly define the given operators Px, Py, and Pz in their matrix forms, "
        "and identify the basis vectors as eigenvectors of Pz. Confirm explicit matrix components and basis context."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining operators and basis, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Confirm and analyze the given system state vector
    cot_instruction_2 = (
        "Sub-task 2: Confirm and analyze the given system state vector, which is the eigenstate of Px with eigenvalue -ħ, "
        "expressed as the column vector (-1/2, 1/sqrt(2), -1/2) in the Pz eigenbasis. Understand its representation."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing system state vector, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Calculate expectation value <Pz>
    cot_instruction_3 = (
        "Sub-task 3: Calculate the expectation value <Pz> of the operator Pz in the given state vector. "
        "Perform matrix multiplication and inner product using Pz matrix and the state vector."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating <Pz>, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Calculate expectation value <Pz^2>
    cot_instruction_4 = (
        "Sub-task 4: Calculate the expectation value <Pz^2> of the operator Pz squared in the given state vector. "
        "Compute Pz^2 by matrix multiplication of Pz with itself, then evaluate expectation value with the state vector."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating <Pz^2>, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 1: Compute uncertainty ΔPz
    cot_instruction_5 = (
        "Sub-task 5: Compute the uncertainty ΔPz of the operator Pz in the given state using the formula "
        "ΔPz = sqrt(<Pz^2> - <Pz>^2), utilizing the previously calculated expectation values."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, computing uncertainty ΔPz, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Compare computed uncertainty ΔPz with multiple-choice options
    debate_instruction_6 = (
        "Sub-task 6: Compare the computed uncertainty ΔPz with the provided multiple-choice options: ħ, ħ/2, ħ/√2, √2 ħ. "
        "Select the correct value that matches the calculated uncertainty."
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing uncertainty with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct uncertainty value ΔPz.", is_sub_task=True)
    agents.append(f"Final Decision agent on uncertainty selection, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
