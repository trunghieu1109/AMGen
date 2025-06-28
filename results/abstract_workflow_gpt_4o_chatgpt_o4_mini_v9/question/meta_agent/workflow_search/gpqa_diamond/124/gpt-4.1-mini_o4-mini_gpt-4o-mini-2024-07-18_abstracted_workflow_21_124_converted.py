async def forward_124(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and Classify Elements

    # Sub-task 1: Analyze the given quantum mechanical system
    cot_instruction_1 = (
        "Sub-task 1: Analyze the quantum system of a particle of mass m in a 3D isotropic harmonic oscillator potential V(r) = (1/2) m ω^2 r^2, "
        "identify the nature of the potential and relevant quantum system characteristics related to energy eigenvalues and eigenfunctions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing quantum system, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Classify quantum states and degeneracy structure
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, classify the quantum states of the 3D isotropic harmonic oscillator in terms of quantum numbers (n, l, m_l), "
        "explain the degeneracy structure of energy levels, and how energy depends on the principal quantum number n."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying quantum states, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most frequent answer for consistency
    from collections import Counter
    answer2_counter = Counter(possible_answers_2)
    answer2_most_common = answer2_counter.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_most_common]
    answer2 = answermapping_2[answer2_most_common]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Generate and Evaluate Variants

    # Sub-task 3: Generate formula for energy eigenvalues and identify quantum number for third excited state
    cot_instruction_3 = (
        "Sub-task 3: Using the classification from Sub-task 2, generate the formula for the energy eigenvalues of the 3D isotropic harmonic oscillator, "
        "express energy of nth excited state in terms of ħω, and identify the quantum number n corresponding to the third excited state."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, generating energy formula, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate degeneracy of the third excited state
    cot_instruction_4 = (
        "Sub-task 4: Based on Sub-task 2, evaluate the degeneracy (number of linearly independent eigenfunctions) corresponding to the energy eigenvalue of the third excited state, "
        "by calculating the number of states with the same energy quantum number."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating degeneracy, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Compute Quantitative or Conditional Measure

    # Sub-task 5: Compute numerical value of energy of third excited state
    cot_instruction_5 = (
        "Sub-task 5: Compute the numerical value of the energy of the third excited state using the formula derived in Sub-task 3, "
        "substituting the appropriate quantum number and constants."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, computing numerical energy, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compute degeneracy for third excited state
    cot_instruction_6 = (
        "Sub-task 6: Compute the degeneracy (number of linearly independent eigenfunctions) for the third excited state using the degeneracy formula or counting method from Sub-task 4."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, computing degeneracy number, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Compare computed energy and degeneracy with given multiple-choice options
    debate_instruction_7 = (
        "Sub-task 7: Compare the computed energy and degeneracy values with the given multiple-choice options to identify the correct answer pair."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent(
                    [taskInfo, thinking5, answer5, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking5, answer5, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing answers, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7(
        [taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct answer pair.", is_sub_task=True)
    agents.append(f"Final Decision agent on final answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
