async def forward_17(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Extract and calculate elemental abundances
    # Sub-task 1: Extract and understand elemental abundance notations and their meaning
    cot_instruction_1 = (
        "Sub-task 1: Extract and understand the elemental abundance notations [Si/Fe]_1, [Mg/Si]_2, [Fe/H]_1, and [Mg/H]_2, "
        "including the meaning of dex units and how these relate to elemental number ratios for Star_1 and Star_2."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding elemental abundance notations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Interpret solar photospheric composition values to calculate baseline number ratios
    cot_instruction_2 = (
        "Sub-task 2: Interpret the solar photospheric composition values 12 + log10(nFe/nH) = 7.5 and 12 + log10(nMg/nH) = 7, "
        "to calculate the baseline number ratios nFe/nH and nMg/nH for the Sun."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, interpreting solar baseline ratios, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate elemental abundances nFe/nH, nMg/nH, nSi/nH for Star_1 and Star_2
    cot_sc_instruction_3 = (
        "Sub-task 3: Using the solar baseline ratios and given abundance notations, calculate the elemental abundances "
        "nFe/nH, nMg/nH, and nSi/nH for Star_1 and Star_2 applying LTE and EW method assumptions."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculating elemental abundances, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most frequent answer for consistency
    answer3_final = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Derive silicon ratios and compare
    # Sub-task 4: Derive nSi/nH for Star_1 and Star_2 by combining abundance ratios and solar baseline
    cot_instruction_4 = (
        "Sub-task 4: Derive the silicon to hydrogen number ratios (nSi/nH) for Star_1 and Star_2 by converting dex abundance ratios "
        "into linear number ratios and combining with solar baseline values."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3_final, answer3_final], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, deriving nSi/nH ratios, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Calculate ratio of silicon atoms in photospheres of Star_1 and Star_2
    cot_instruction_5 = (
        "Sub-task 5: Calculate the ratio of silicon atoms in the photospheres of Star_1 and Star_2 by dividing their respective nSi/nH values."
    )
    debate_roles = ["Agent A", "Agent B"]
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
            agents.append(f"Debate agent {agent.id}, round {r}, calculating silicon atom ratio, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on silicon atom ratio.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating silicon atom ratio, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare calculated silicon atom ratio to multiple-choice options and identify closest match
    cot_instruction_6 = (
        "Sub-task 6: Compare the calculated silicon atom ratio to the provided options (~0.8, ~12.6, ~3.9, ~1.2) "
        "and identify the closest match as the final answer."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing silicon ratio to choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
