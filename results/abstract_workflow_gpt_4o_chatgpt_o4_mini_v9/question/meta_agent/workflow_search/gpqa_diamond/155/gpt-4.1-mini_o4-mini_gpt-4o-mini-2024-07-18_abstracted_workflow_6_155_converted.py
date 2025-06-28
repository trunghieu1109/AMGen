async def forward_155(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze starting materials and reaction mechanisms
    
    # Sub-task 1: Analyze chemical nature and stereochemistry of (E)- and (Z)-oct-4-ene
    cot_instruction_1 = (
        "Sub-task 1: Analyze the chemical nature and stereochemistry of the starting materials: "
        "(E)-oct-4-ene and (Z)-oct-4-ene, including their geometric isomerism and implications for reaction outcomes."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing starting materials, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine reaction mechanism and product stereochemistry for (E)-oct-4-ene with mCPBA and aqueous acid
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1, determine the reaction mechanism and product stereochemistry when (E)-oct-4-ene is treated with one equivalent of mCPBA followed by aqueous acid, "
        "considering stereochemical outcome of epoxidation and ring opening."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing (E)-oct-4-ene reaction, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose most consistent answer
    from collections import Counter
    counter_2 = Counter(possible_answers_2)
    answer2_final = counter_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Determine reaction mechanism and product stereochemistry for (Z)-oct-4-ene with mCPBA and aqueous acid
    cot_sc_instruction_3 = (
        "Sub-task 3: Based on Sub-task 1, determine the reaction mechanism and product stereochemistry when (Z)-oct-4-ene is treated with one equivalent of mCPBA followed by aqueous acid, "
        "considering stereochemical outcome of epoxidation and ring opening."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing (Z)-oct-4-ene reaction, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    counter_3 = Counter(possible_answers_3)
    answer3_final = counter_3.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Identify and enumerate distinct stereoisomeric products from both reactions combined
    cot_reflect_instruction_4 = (
        "Sub-task 4: Based on Sub-tasks 2 and 3, identify and enumerate the distinct stereoisomeric products formed from both reactions combined, "
        "considering stereochemical analysis."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking2_final, answermapping_2[answer2_final], thinking3_final, answermapping_3[answer3_final]]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, enumerating stereoisomers, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                               "Critically evaluate the enumeration of stereoisomers for completeness and correctness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback on stereoisomer enumeration, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining stereoisomer enumeration, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Analyze chromatographic behavior

    # Sub-task 5: Analyze expected chromatographic behavior on standard (achiral) reverse-phase HPLC
    debate_instruction_5 = (
        "Sub-task 5: Based on Sub-task 4 output, analyze the expected chromatographic behavior of the combined product mixture on a standard (achiral) reverse-phase HPLC column, "
        "predicting the number of resolved peaks based on the number and nature of stereoisomers identified."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            input_infos_5 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos_5 += all_thinking5[r-1] + all_answer5[r-1]
            thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing standard HPLC peaks, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1],
                                                    "Sub-task 5: Make final decision on number of peaks in standard HPLC.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent on standard HPLC peaks, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Analyze expected chromatographic behavior on chiral HPLC
    debate_instruction_6 = (
        "Sub-task 6: Based on Sub-task 4 output, analyze the expected chromatographic behavior of the combined product mixture on a chiral HPLC column, "
        "predicting the number of resolved peaks based on the number and nature of stereoisomers and their chiral properties."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            input_infos_6 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos_6 += all_thinking6[r-1] + all_answer6[r-1]
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing chiral HPLC peaks, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1],
                                                    "Sub-task 6: Make final decision on number of peaks in chiral HPLC.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent on chiral HPLC peaks, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Compare and summarize chromatographic outcomes
    cot_reflect_instruction_7 = (
        "Sub-task 7: Compare and summarize the chromatographic outcomes from the standard and chiral HPLC analyses to determine the final observed number of peaks in each chromatogram, "
        "considering theoretical maximum resolution."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking5, answer5, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, summarizing chromatographic outcomes, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7],
                                               "Critically evaluate the chromatographic summary for accuracy and completeness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, feedback on chromatographic summary, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining chromatographic summary, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
