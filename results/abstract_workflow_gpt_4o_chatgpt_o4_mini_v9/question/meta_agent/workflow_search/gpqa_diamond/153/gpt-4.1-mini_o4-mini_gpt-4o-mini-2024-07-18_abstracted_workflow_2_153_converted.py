async def forward_153(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the given spectral data (mass spectrum, IR spectrum, and 1H NMR spectrum)
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given spectral data (mass spectrum, IR spectrum, and 1H NMR spectrum) "
        "to identify and classify key features such as molecular weight, isotopic pattern, functional groups, "
        "and proton environments relevant to the unknown compound."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing spectral data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Interpret the mass spectrum data
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, interpret the mass spectrum data, focusing on the molecular ion peak at m/z = 156 (100%) "
        "and the isotopic peak at m/z = 158 (32%), to deduce the molecular formula or presence of specific elements such as chlorine."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, interpreting mass spectrum, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote)
    answer2_final = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze the IR spectrum data
    cot_instruction_3 = (
        "Sub-task 3: Analyze the IR spectrum data, specifically the broad peak from 3500-2700 cm^-1 and the strong sharp peak at 1720 cm^-1, "
        "to identify the functional groups present in the compound such as carboxylic acid, aldehyde, or ester."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing IR spectrum, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Interpret the 1H NMR spectrum data
    cot_instruction_4 = (
        "Sub-task 4: Interpret the 1H NMR spectrum data, including chemical shifts at 11.0 ppm (singlet, 1H), 8.02 ppm (doublet, 2H), and 7.72 ppm (doublet, 2H), "
        "to deduce the proton environments and possible aromatic substitution patterns in the compound."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, interpreting 1H NMR spectrum, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Integrate and Propose Structural Candidates

    # Sub-task 5: Integrate mass spectral data with IR functional group info
    debate_instruction_5 = (
        "Sub-task 5: Integrate the interpreted mass spectral data (molecular weight and isotopic pattern) "
        "with the functional group information from IR analysis to narrow down possible structural classes of the compound."
    )
    debate_roles = ["Mass Spectrum Specialist", "IR Spectrum Specialist"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking2_final, answer2_final, thinking3, answer3], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking2_final, answer2_final, thinking3, answer3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating mass and IR data, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on integrated mass and IR data.", is_sub_task=True)
    agents.append(f"Final Decision agent, integrating mass and IR data, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Combine 1H NMR data with integrated mass and IR data to propose structural candidates
    cot_instruction_6 = (
        "Sub-task 6: Combine the proton environment and aromatic substitution pattern information from 1H NMR analysis "
        "with the integrated mass and IR data to propose plausible structural candidates for the unknown compound."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, proposing structural candidates, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Compare proposed candidates against multiple-choice options
    cot_instruction_7 = (
        "Sub-task 7: Compare the proposed structural candidates against the provided multiple-choice options: "
        "4-chlorobenzoic acid, 2-chlorobenzoic acid, 3-chloro-2-hydroxybenzaldehyde, phenyl chloroformate, "
        "to select the most reasonable structural suggestion for the unidentified drug."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, selecting final structural suggestion, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
