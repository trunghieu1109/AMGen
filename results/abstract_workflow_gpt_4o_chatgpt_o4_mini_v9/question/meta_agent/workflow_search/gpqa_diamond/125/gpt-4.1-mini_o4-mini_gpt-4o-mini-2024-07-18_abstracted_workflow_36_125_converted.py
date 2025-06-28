async def forward_125(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Identify starting materials and chromatographic techniques
    # Sub-task 1: Identify and characterize starting materials and reagents
    cot_instruction_1 = (
        "Sub-task 1: Identify and characterize the starting materials and reagents used in both reactions, "
        "including stereochemistry and functional groups, to understand the initial chemical entities involved."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying starting materials and reagents, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Extract and define chromatographic techniques
    cot_instruction_2 = (
        "Sub-task 2: Extract and define the chromatographic techniques used (TLC, normal-phase HPLC, chiral HPLC) "
        "and their principles relevant to separation of reaction products, including how chiral stationary phases separate enantiomers."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, defining chromatographic techniques, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Apply reaction conditions and assess stereochemical outcomes and reaction completeness
    # Sub-task 3: Predict products after reduction
    cot_instruction_3 = (
        "Sub-task 3: Apply the reaction conditions to the starting materials: (S)-5-methoxyhexan-3-one treated with LAH and acidic workup, "
        "and pentane-2,4-dione treated with excess NaBH4 and acidic workup, to predict the chemical structures of the products formed after reduction."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, predicting products after reduction, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Assess stereochemical outcomes
    cot_instruction_4 = (
        "Sub-task 4: Assess the stereochemical outcomes of each reaction product, including whether new stereocenters are formed and if the products are chiral or achiral, "
        "to understand the number and type of stereoisomers present."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, assessing stereochemical outcomes, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Validate reaction completeness
    debate_instruction_5 = (
        "Sub-task 5: Validate the completeness of the reactions (100% conversion) and confirm that no side products or incomplete reactions remain, "
        "ensuring the product mixture consists only of the predicted products."
    )
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating reaction completeness, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on reaction completeness.", is_sub_task=True)
    agents.append(f"Final Decision agent, validating reaction completeness, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Analyze combined product mixture and predict chromatogram peaks
    # Sub-task 6: Analyze combined product mixture for distinct species
    cot_instruction_6 = (
        "Sub-task 6: Analyze the combined product mixture to determine the number of distinct chemical species present, considering stereoisomers and structural isomers, "
        "that would be separated by chromatographic methods."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, analyzing combined product mixture, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Predict number of peaks in normal-phase HPLC
    cot_sc_instruction_7 = (
        "Sub-task 7: Classify and predict the number of peaks expected in the normal-phase HPLC chromatogram based on the chemical and stereochemical differences of the combined products, "
        "considering that normal-phase HPLC separates mainly by polarity and structural differences but not enantiomers."
    )
    N_sc = self.max_sc
    cot_agents_7 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]
    possible_answers_7 = []
    thinkingmapping_7 = {}
    answermapping_7 = {}
    for i in range(N_sc):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking2, answer2, thinking6, answer6], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, predicting normal-phase HPLC peaks, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7.content)
        thinkingmapping_7[answer7.content] = thinking7
        answermapping_7[answer7.content] = answer7
    # Choose most frequent answer
    count_7 = Counter(possible_answers_7)
    normal_phase_peaks = count_7.most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinkingmapping_7[normal_phase_peaks].content}; answer - {normal_phase_peaks}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Predict number of peaks in chiral stationary phase HPLC
    cot_sc_instruction_8 = (
        "Sub-task 8: Classify and predict the number of peaks expected in the chiral stationary phase HPLC chromatogram based on the presence of enantiomers or diastereomers in the combined product mixture, "
        "considering that chiral HPLC separates enantiomers into distinct peaks."
    )
    cot_agents_8 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]
    possible_answers_8 = []
    thinkingmapping_8 = {}
    answermapping_8 = {}
    for i in range(N_sc):
        thinking8, answer8 = await cot_agents_8[i]([taskInfo, thinking2, answer2, thinking6, answer6], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_8[i].id}, predicting chiral HPLC peaks, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers_8.append(answer8.content)
        thinkingmapping_8[answer8.content] = thinking8
        answermapping_8[answer8.content] = answer8
    count_8 = Counter(possible_answers_8)
    chiral_phase_peaks = count_8.most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinkingmapping_8[chiral_phase_peaks].content}; answer - {chiral_phase_peaks}")
    print("Step 8: ", sub_tasks[-1])

    # Final answer synthesis
    final_answer = await self.make_final_answer(
        thinking8,
        answer8,
        sub_tasks,
        agents
    )
    return final_answer
