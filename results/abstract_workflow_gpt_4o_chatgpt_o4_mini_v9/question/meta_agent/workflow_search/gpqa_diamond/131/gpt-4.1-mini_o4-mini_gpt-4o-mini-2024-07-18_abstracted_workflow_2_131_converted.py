async def forward_131(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and characterize spectral data and candidate compounds

    # Sub-task 1: Analyze the given 1H NMR spectral data of the mixture
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given 1H NMR spectral data of the mixture: identify and characterize the signals at ~6.7 ppm "
        "(two singlets, 1:1 ratio) and at ~2.2 ppm (three singlets, 2:1:1 ratio), including their chemical shift, multiplicity, and integration ratios. "
        "Provide detailed baseline spectral features to be explained by the compounds."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing 1H NMR spectral data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze molecular formula C10H14 and structural features of each candidate compound
    cot_instruction_2 = (
        "Sub-task 2: Analyze the molecular formula C10H14 and the structural features of each candidate compound: "
        "1,2,4,5-tetramethylbenzene, 1,2,3,5-tetramethylbenzene, 1,2,3,4-tetramethylbenzene, and 1,4-diethylbenzene. "
        "Focus on the number and types of aromatic protons and alkyl substituents that influence their 1H NMR spectra."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing molecular formula and structures, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Predict expected 1H NMR aromatic region signals for each candidate compound
    cot_sc_instruction_3 = (
        "Sub-task 3: Based on the structural analysis from Sub-task 2, predict the expected 1H NMR aromatic region signals "
        "(chemical shifts, multiplicities, and relative integrations) for each candidate compound, considering substitution pattern and symmetry."
    )
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, predicting aromatic signals, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer (majority vote)
    answer3_final = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Predict expected 1H NMR aliphatic region signals for each candidate compound
    cot_sc_instruction_4 = (
        "Sub-task 4: Based on the structural analysis from Sub-task 2, predict the expected 1H NMR aliphatic region signals "
        "(chemical shifts, multiplicities, and relative integrations) for each candidate compound, focusing on methyl or ethyl groups and their chemical environment and symmetry."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, predicting aliphatic signals, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_final = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4_final = thinkingmapping_4[answer4_final]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Compare predicted signals with observed signals and integrate results

    # Sub-task 5: Compare predicted aromatic proton signals with observed aromatic signals
    debate_instruction_5 = (
        "Sub-task 5: Compare the predicted aromatic proton signals of each candidate compound with the observed aromatic signals "
        "(two singlets at ~6.7 ppm in 1:1 ratio) to identify which pairs of compounds can produce exactly two aromatic singlets in a 1:1 ratio when mixed 1:1."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3_final, answer3_final], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3_final, answer3_final] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing aromatic signals, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on which pairs match aromatic signals.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding aromatic match, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare predicted aliphatic proton signals with observed aliphatic signals
    debate_instruction_6 = (
        "Sub-task 6: Compare the predicted aliphatic proton signals of each candidate compound with the observed aliphatic signals "
        "(three singlets at ~2.2 ppm in 2:1:1 ratio) to identify which pairs of compounds can produce three singlets in the observed integration ratio when mixed 1:1."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    all_thinking6 = [[] for _ in range(N_max_5)]
    all_answer6 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4_final, answer4_final], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking4_final, answer4_final] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing aliphatic signals, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on which pairs match aliphatic signals.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding aliphatic match, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Integrate results from aromatic and aliphatic comparisons to determine best matching pair
    cot_reflect_instruction_7 = (
        "Sub-task 7: Integrate the results from Sub-task 5 and Sub-task 6 to determine which pair of compounds from the given options "
        "best matches both the aromatic and aliphatic 1H NMR spectral features observed in the mixture. Provide a reasoned conclusion."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_7 = [taskInfo, thinking5, answer5, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, integrating aromatic and aliphatic results, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
