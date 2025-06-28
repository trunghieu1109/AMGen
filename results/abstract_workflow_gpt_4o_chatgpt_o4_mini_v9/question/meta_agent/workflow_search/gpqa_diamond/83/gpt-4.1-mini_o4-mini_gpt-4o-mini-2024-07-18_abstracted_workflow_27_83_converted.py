async def forward_83(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Understand context and analyze fractional approximation and parallelization factors

    # Sub-task 1: Understand the context of solving higher dimensional heat equations with fractional approximation
    cot_instruction_1 = (
        "Sub-task 1: Understand the context of solving higher dimensional heat equations using higher order finite difference approximations and parallel splitting, "
        "including the role of the matrix exponential function and its fractional approximation."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding context, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze fractional approximation of matrix exponential function
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, analyze the concept of fractional approximation of the matrix exponential function, "
        "focusing on its mathematical properties and usage in numerical PDE methods."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing fractional approximation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote)
    counter_2 = Counter(possible_answers_2)
    answer2_final = counter_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify key factors in converting sequential to parallel algorithm
    cot_instruction_3 = (
        "Sub-task 3: Identify and explain the key factors involved in converting a sequential algorithm into a parallel algorithm in the context of numerical PDE solving, "
        "especially when fractional approximations are used, based on previous outputs."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2_final, answermapping_2[answer2_final]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, identifying key factors, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Examine each provided choice in relation to parallelization relevance
    cot_instruction_4 = (
        "Sub-task 4: Examine each of the provided choices (Existence of nonlocal boundary conditions, Stability analysis, "
        "Linear partial fraction of fractional approximation, Complex roots of fractional approximation) in relation to their relevance and role in parallelizing the algorithm for solving heat equations with fractional approximations."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, examining choices relevance, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Evaluate importance of each factor and prioritize

    # Sub-task 5: Evaluate importance of stability analysis
    cot_instruction_5 = (
        "Sub-task 5: Evaluate the importance of stability analysis in the parallelization of algorithms for solving higher dimensional heat equations with fractional approximations, "
        "based on previous understanding."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, evaluating stability analysis, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Assess role of linear partial fraction decomposition
    cot_instruction_6 = (
        "Sub-task 6: Assess the role of linear partial fraction decomposition of fractional approximations in enabling parallel splitting and converting sequential algorithms into parallel ones."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, assessing linear partial fraction role, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Analyze impact of complex roots of fractional approximations
    cot_instruction_7 = (
        "Sub-task 7: Analyze the impact of complex roots of fractional approximations on the parallelization process and whether they are a key factor in converting sequential algorithms to parallel algorithms."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking4, answer4], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, analyzing complex roots impact, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Determine relevance of nonlocal boundary conditions
    cot_instruction_8 = (
        "Sub-task 8: Determine the relevance of nonlocal boundary conditions to the parallelization of the algorithm and whether their existence is a key factor in converting sequential algorithms into parallel algorithms."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking4, answer4], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, determining nonlocal boundary conditions relevance, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    # Sub-task 9: Compare and prioritize the four choices to identify the key factor
    debate_instruction_9 = (
        "Sub-task 9: Compare and prioritize the four choices (Existence of nonlocal boundary conditions, Stability analysis, "
        "Linear partial fraction of fractional approximation, Complex roots of fractional approximation) based on their evaluated roles and importance in the parallelization of the algorithm, "
        "to identify the key factor that enables conversion from sequential to parallel algorithm."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]

    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                input_infos_9 = [taskInfo, thinking5, answer5, thinking6, answer6, thinking7, answer7, thinking8, answer8]
            else:
                input_infos_9 = [taskInfo, thinking5, answer5, thinking6, answer6, thinking7, answer7, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
            thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, prioritizing key factor, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)

    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the key factor enabling parallelization.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding key factor, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer
