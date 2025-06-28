async def forward_131(self, taskInfo):
    from collections import Counter

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 0: Extract and characterize defining features of the given 1H NMR spectrum data and candidate compounds
    # Sub-task 1: Extract and characterize the defining features of the given 1H NMR spectrum data
    cot_instruction_0_1 = (
        "Sub-task 1: Extract and characterize the defining features of the given 1H NMR spectrum data: "
        "identify the number of signals, their chemical shifts, multiplicities, and integration ratios, "
        "specifically focusing on the aromatic region (~6.7 ppm) and the aliphatic region (~2.2 ppm) for the 1:1 mixture of two C10H14 aromatic compounds."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracting NMR spectrum features, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Extract and characterize the structural features of each candidate compound
    cot_instruction_0_2 = (
        "Sub-task 2: Extract and characterize the structural features of each candidate compound (1,2,4,5-tetramethylbenzene, "
        "1,2,3,5-tetramethylbenzene, 1,2,3,4-tetramethylbenzene, and 1,4-diethylbenzene), focusing on the number and types of chemically distinct protons "
        "(aromatic and aliphatic), expected chemical shifts, and splitting patterns relevant to 1H NMR."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, extracting candidate compound features, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Analyze aromatic and aliphatic proton environments of each candidate compound
    # Sub-task 3: Analyze aromatic proton environments to predict number of aromatic signals, multiplicities, and integration ratios
    cot_sc_instruction_1_3 = (
        "Sub-task 3: Analyze the aromatic proton environments of each candidate compound based on their substitution patterns to predict "
        "the number of aromatic signals, their multiplicities (singlet, doublet, etc.), and relative integration ratios, to compare with the observed two singlets at ~6.7 ppm in a 1:1 ratio."
    )
    N_sc = self.max_sc
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_3 = []
    thinkingmapping_1_3 = {}
    answermapping_1_3 = {}
    for i in range(N_sc):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, analyzing aromatic protons, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3.content)
        thinkingmapping_1_3[answer_1_3.content] = thinking_1_3
        answermapping_1_3[answer_1_3.content] = answer_1_3
    most_common_answer_1_3 = Counter(possible_answers_1_3).most_common(1)[0][0]
    thinking_1_3 = thinkingmapping_1_3[most_common_answer_1_3]
    answer_1_3 = answermapping_1_3[most_common_answer_1_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Analyze aliphatic proton environments to predict number of aliphatic signals, multiplicities, and integration ratios
    cot_sc_instruction_1_4 = (
        "Sub-task 4: Analyze the aliphatic proton environments of each candidate compound to predict the number of aliphatic signals, "
        "their multiplicities (singlets), and relative integration ratios, to compare with the observed three singlets at ~2.2 ppm in a 2:1:1 ratio."
    )
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_4 = []
    thinkingmapping_1_4 = {}
    answermapping_1_4 = {}
    for i in range(N_sc):
        thinking_1_4, answer_1_4 = await cot_agents_1_4[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, analyzing aliphatic protons, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
        possible_answers_1_4.append(answer_1_4.content)
        thinkingmapping_1_4[answer_1_4.content] = thinking_1_4
        answermapping_1_4[answer_1_4.content] = answer_1_4
    most_common_answer_1_4 = Counter(possible_answers_1_4).most_common(1)[0][0]
    thinking_1_4 = thinkingmapping_1_4[most_common_answer_1_4]
    answer_1_4 = answermapping_1_4[most_common_answer_1_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Combine predicted aromatic and aliphatic NMR features of pairs of candidate compounds
    # Sub-task 5: Combine predicted features to simulate expected combined 1H NMR spectrum of 1:1 mixture
    debate_instruction_2_5 = (
        "Sub-task 5: Combine the predicted aromatic and aliphatic NMR features of pairs of candidate compounds (from the four given options) "
        "to simulate the expected combined 1H NMR spectrum of a 1:1 mixture, focusing on the number of signals, their chemical shifts, multiplicities, "
        "and integration ratios, to match the observed spectrum."
    )
    debate_agents_2_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_5 = self.max_round
    all_thinking_2_5 = [[] for _ in range(N_max_2_5)]
    all_answer_2_5 = [[] for _ in range(N_max_2_5)]
    for r in range(N_max_2_5):
        for i, agent in enumerate(debate_agents_2_5):
            if r == 0:
                input_infos_2_5 = [taskInfo, thinking_1_3, answer_1_3, thinking_1_4, answer_1_4]
            else:
                input_infos_2_5 = [taskInfo, thinking_1_3, answer_1_3, thinking_1_4, answer_1_4] + all_thinking_2_5[r-1] + all_answer_2_5[r-1]
            thinking_2_5, answer_2_5 = await agent(input_infos_2_5, debate_instruction_2_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, combining predicted NMR features, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
            all_thinking_2_5[r].append(thinking_2_5)
            all_answer_2_5[r].append(answer_2_5)
    final_decision_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_5, answer_2_5 = await final_decision_agent_2_5([taskInfo] + all_thinking_2_5[-1] + all_answer_2_5[-1], "Sub-task 5: Make final decision on the combined predicted NMR spectra of candidate pairs.", is_sub_task=True)
    agents.append(f"Final Decision agent on combined NMR features, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Stage 3: Evaluate and prioritize candidate pairs by comparing simulated combined spectra against observed spectrum
    # Sub-task 6: Evaluate and prioritize candidate pairs to identify best fit
    debate_instruction_3_6 = (
        "Sub-task 6: Evaluate and prioritize the candidate pairs by comparing their simulated combined NMR spectra against the observed spectrum, "
        "specifically checking for the presence of exactly two aromatic singlets at ~6.7 ppm in a 1:1 ratio and three aliphatic singlets at ~2.2 ppm in a 2:1:1 ratio, "
        "to identify which pair best fits the experimental data."
    )
    debate_agents_3_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_6 = self.max_round
    all_thinking_3_6 = [[] for _ in range(N_max_3_6)]
    all_answer_3_6 = [[] for _ in range(N_max_3_6)]
    for r in range(N_max_3_6):
        for i, agent in enumerate(debate_agents_3_6):
            if r == 0:
                input_infos_3_6 = [taskInfo, thinking_2_5, answer_2_5]
            else:
                input_infos_3_6 = [taskInfo, thinking_2_5, answer_2_5] + all_thinking_3_6[r-1] + all_answer_3_6[r-1]
            thinking_3_6, answer_3_6 = await agent(input_infos_3_6, debate_instruction_3_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating and prioritizing candidate pairs, thinking: {thinking_3_6.content}; answer: {answer_3_6.content}")
            all_thinking_3_6[r].append(thinking_3_6)
            all_answer_3_6[r].append(answer_3_6)
    final_decision_agent_3_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_6, answer_3_6 = await final_decision_agent_3_6([taskInfo] + all_thinking_3_6[-1] + all_answer_3_6[-1], "Sub-task 6: Make final decision on the best fitting candidate pair.", is_sub_task=True)
    agents.append(f"Final Decision agent on prioritizing candidate pairs, thinking: {thinking_3_6.content}; answer: {answer_3_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_3_6.content}; answer - {answer_3_6.content}")
    print("Step 3.6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_6, answer_3_6, sub_tasks, agents)
    return final_answer
