async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze the given chemical system
    cot_instruction_1 = (
        "Sub-task 1: Analyze the chemical system containing Co(II) ions and SCN- ions, "
        "identify all species involved (free Co(II), free SCN-, and complexes with 1 to 4 SCN- ligands), "
        "their initial concentrations (c(Co) = 10^-2 M, [SCN-] = 0.1 M), and the given stability constants β1=9, β2=40, β3=63, β4=16. "
        "Classify the complexes by their stoichiometry: mono-, di-, tri-, and tetrathiocyanato cobalt(II) complexes."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing chemical system, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Generate equilibrium expressions and evaluate balances
    cot_sc_instruction_2 = (
        "Sub-task 2: Using the output from Sub-task 1, generate equilibrium expressions for the formation of each cobalt(II) thiocyanato complex "
        "based on the stability constants β1=9, β2=40, β3=63, β4=16. Express the concentration of each complex in terms of free Co(II) and free SCN- concentrations."
    )
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, generating equilibrium expressions, thinking: {thinking2.content}; answer: {answer2.content}")
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

    # Sub-task 3: Evaluate total cobalt balance equation
    cot_instruction_3 = (
        "Sub-task 3: Using the equilibrium expressions from Sub-task 2, evaluate the total cobalt balance equation by summing concentrations of all cobalt species "
        "(free Co(II) plus all complexes) and relate it to the total cobalt concentration (10^-2 M)."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, evaluating cobalt balance, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate free SCN- concentration approximation
    cot_instruction_4 = (
        "Sub-task 4: Determine if the free SCN- concentration can be approximated as the initial concentration (0.1 M) or if it needs adjustment based on complex formation, using outputs from Sub-task 2."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating free SCN- concentration, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Compute concentrations and percentage of dithiocyanato complex
    debate_instruction_5 = (
        "Sub-task 5: Based on the outputs from Sub-tasks 3 and 4, solve the system of equations to compute concentrations of each cobalt(II) thiocyanato complex species. "
        "Then calculate the percentage of the blue dithiocyanato cobalt(II) complex (with two SCN- ligands) relative to total cobalt concentration."
    )
    debate_roles = getattr(self, 'debate_role', ['Agent A', 'Agent B'])
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing concentrations and percentage, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the percentage of the blue dithiocyanato cobalt(II) complex.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final percentage, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
