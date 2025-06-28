async def forward_2(self, taskInfo):
    from collections import Counter
    import math

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Identify Elements

    # Sub-task 1: Analyze and identify the given quantum state and basis states
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given quantum state 0.5|\u2191\u27e9 + sqrt(3)/2|\u2193\u27e9, "
        "and confirm that |\u2191\u27e9 and |\u2193\u27e9 are eigenstates of the Pauli sigma_z operator. "
        "Explain the meaning of these basis states and their relation to sigma_z eigenvalues."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing quantum state and basis states, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Analyze and identify the operator 10sigma_z + 5sigma_x and Pauli matrices
    cot_instruction_2 = (
        "Sub-task 2: Analyze the operator 10\u03c3_z + 5\u03c3_x, including the meaning and matrix representations "
        "of the Pauli matrices \u03c3_z and \u03c3_x in the |\u2191\u27e9, |\u2193\u27e9 basis. "
        "Explain how these matrices act on the basis states."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing operator and Pauli matrices, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Express State and Operator in Matrix Form

    # Sub-task 3: Express the quantum state as a column vector
    cot_instruction_3 = (
        "Sub-task 3: Express the given quantum state as a column vector in the |\u2191\u27e9, |\u2193\u27e9 basis "
        "using coefficients 0.5 and sqrt(3)/2, preparing it for matrix operations."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, expressing quantum state as vector, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Write down the matrix form of the operator 10sigma_z + 5sigma_x
    cot_instruction_4 = (
        "Sub-task 4: Write the matrix form of the operator 10\u03c3_z + 5\u03c3_x using standard Pauli matrices "
        "in the |\u2191\u27e9, |\u2193\u27e9 basis."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, writing operator matrix, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 3: Calculate Expectation Value and Select Correct Answer

    # Sub-task 5: Calculate expectation value <psi| (10sigma_z + 5sigma_x) |psi>
    debate_instruction_5 = (
        "Sub-task 5: Calculate the expectation value of the operator 10\u03c3_z + 5\u03c3_x in the given quantum state "
        "by performing the matrix multiplication <\u03c8| (10\u03c3_z + 5\u03c3_x) |\u03c8>, where |\u03c8\u27e9 is the state vector from Sub-task 3 and the operator matrix from Sub-task 4. "
        "Provide detailed reasoning and final numeric result."
    )
    debate_roles = ["Agent A", "Agent B"]
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
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating expectation value, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Make final decision on the calculated expectation value and select the correct answer from the choices 0.85, 1.65, -1.4, -0.7.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, calculating final expectation value and selecting answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Round the expectation value and compare with choices
    cot_instruction_6 = (
        "Sub-task 6: Round the calculated expectation value to one decimal place and compare it with the provided multiple-choice options: 0.85, 1.65, -1.4, -0.7. "
        "Select and justify the correct answer."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, rounding and selecting final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
