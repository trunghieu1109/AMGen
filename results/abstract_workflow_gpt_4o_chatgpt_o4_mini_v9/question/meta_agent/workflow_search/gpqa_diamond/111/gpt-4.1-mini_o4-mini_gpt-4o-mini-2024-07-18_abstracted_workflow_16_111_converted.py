async def forward_111(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Normalize state vector and write operator matrix

    # Sub-task 1: Normalize the given state |alpha> = (1+i)|up> + (2-i)|down>
    cot_instruction_1 = (
        "Sub-task 1: Normalize the quantum state |alpha> = (1+i)|up> + (2-i)|down>. "
        "Calculate the normalization factor and express the normalized state vector explicitly. "
        "This is necessary for correct probability and expectation calculations.")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, normalizing state vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Write down matrix representation of operator A
    cot_instruction_2 = (
        "Sub-task 2: Write the 2x2 matrix representation of operator A in the |up>, |down> basis, "
        "where Aij = hbar/2 if i != j, and 0 otherwise. "
        "This prepares the operator for eigenvalue and eigenvector calculations.")
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, writing operator matrix, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Find eigenvalues/eigenvectors and express normalized state in eigenbasis

    # Sub-task 3: Find eigenvalues and eigenvectors of operator A
    cot_sc_instruction_3 = (
        "Sub-task 3: Find eigenvalues and eigenvectors of operator A from Sub-task 2. "
        "This is essential to express the measurement basis and calculate measurement probabilities.")
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, finding eigenvalues/vectors, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer (majority vote)
    counter_3 = Counter(possible_answers_3)
    best_answer_3 = counter_3.most_common(1)[0][0]
    thinking3 = thinkingmapping_3[best_answer_3]
    answer3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Express normalized state |alpha> in eigenbasis of A
    cot_instruction_4 = (
        "Sub-task 4: Express the normalized state |alpha> from Sub-task 1 in the eigenbasis of operator A from Sub-task 3. "
        "Calculate the probability amplitudes for each eigenstate.")
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, expressing normalized state in eigenbasis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Calculate probabilities and expectation value

    # Sub-task 5: Calculate measurement probabilities for each eigenstate
    cot_instruction_5 = (
        "Sub-task 5: Calculate the probability of measuring the particle in each eigenstate of operator A by taking the squared magnitude of the projection of |alpha> onto each eigenvector from Sub-task 4.")
    debate_roles = ["Pro", "Con"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], cot_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, cot_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating measurement probabilities, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on measurement probabilities.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating measurement probabilities, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Calculate expectation value <alpha|A|alpha>
    cot_instruction_6 = (
        "Sub-task 6: Calculate the average (expectation) value of operator A in the normalized state |alpha> by computing <alpha|A|alpha> using the operator matrix from Sub-task 2 and normalized state vector from Sub-task 1.")
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating expectation value, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Stage 4: Compare results with given multiple-choice options

    # Sub-task 7: Compare calculated probabilities and expectation value with choices
    cot_instruction_7 = (
        "Sub-task 7: Compare the calculated probabilities from Sub-task 5 and expectation value from Sub-task 6 "
        "with the given multiple-choice options to identify the correct answer.")
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, comparing results with choices, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
