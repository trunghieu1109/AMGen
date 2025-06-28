async def forward_34(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and formulate the operator and eigenvalue equation
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given operator Ay = c * S, where c = h/(4π) and S is a 2x2 matrix with rows (0, -i) and (i, 0). "
        "Understand the structure and properties of S and the constant c, and interpret the physical meaning of Ay as the Y-component of intrinsic angular momentum operator for a muon."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing operator Ay, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Formulate the eigenvalue equation Ay(φ) = a(φ) explicitly in matrix form
    cot_instruction_2 = (
        "Sub-task 2: Using the operator Ay and its matrix form from Sub-task 1, formulate the eigenvalue equation Ay(φ) = a(φ) explicitly. "
        "Express it in matrix form to prepare for eigenvalue and eigenvector calculation."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formulating eigenvalue equation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Calculate eigenvalues and eigenvectors, analyze implications, and evaluate statements

    # Sub-task 3: Calculate eigenvalues and eigenvectors
    cot_sc_instruction_3 = (
        "Sub-task 3: Calculate the eigenvalues and eigenvectors of the matrix Ay by solving the characteristic equation det(Ay - aI) = 0, "
        "using the matrix form derived in Sub-task 2. Interpret the nature (real/imaginary) and values of the eigenvalues."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculating eigenvalues and eigenvectors, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Aggregate most consistent answer
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinkingmapping_3[most_common_answer_3].content}; answer - {most_common_answer_3}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Analyze physical and mathematical implications of eigenvalues and eigenvectors
    cot_instruction_4 = (
        "Sub-task 4: Analyze the physical and mathematical implications of the eigenvalues and eigenvectors obtained in Sub-task 3, "
        "specifically regarding their relation to the eigenfunctions of A^2 and Az operators, and the basis functions of Ay."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinkingmapping_3[most_common_answer_3], answermapping_3[most_common_answer_3]], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing implications of eigenvalues and eigenvectors, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Evaluate the provided statements against results
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, evaluate each of the provided statements (choice1 to choice4) in the query against the results and interpretations, "
        "to determine which statement(s) correctly describe the eigenvalues, eigenvectors, and their physical significance."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating statements, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on which statements are correct.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on correct statements, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
