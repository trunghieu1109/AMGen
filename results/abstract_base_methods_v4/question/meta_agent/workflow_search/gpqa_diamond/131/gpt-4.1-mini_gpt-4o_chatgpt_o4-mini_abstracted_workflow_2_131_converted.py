async def forward_131(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the molecular formula C10H14 to determine the degree of unsaturation and confirm the presence of aromatic rings and alkyl substituents relevant to the candidate compounds."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular formula C10H14, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction_2a = "Sub-task 2a: Perform a detailed symmetry-element analysis for each candidate compound (1,2,4,5-tetramethylbenzene, 1,2,3,5-tetramethylbenzene, 1,2,3,4-tetramethylbenzene, and 1,4-diethylbenzene) to explicitly identify unique proton environments, including aromatic and aliphatic protons, using structural diagrams and group theory concepts. Emphasize identification of mirror planes, rotational axes, and equivalence classes."
    N2a = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2a)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2a):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1], cot_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, performing symmetry-element analysis, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    cot_instruction_2b = "Sub-task 2b: Based on the symmetry analysis from subtask_2a, list and classify the equivalence classes of aromatic and aliphatic protons for each candidate compound, specifying the expected number of distinct signals and their relative proton counts."
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "CoT"
    }
    thinking2b, answer2b = await cot_agent_2b([taskInfo, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, classifying equivalence classes of protons, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    cot_instruction_3 = "Sub-task 3: Predict the expected 1H NMR aromatic region signals (chemical shifts, multiplicity, and integration) for each candidate compound using the equivalence classes identified in subtask_2b, considering typical chemical shift ranges for aromatic protons and substituent effects."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2b, answer2b], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, predicting aromatic NMR signals, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction_4 = "Sub-task 4: Predict the expected 1H NMR aliphatic region signals (chemical shifts, multiplicity, and integration) for each candidate compound using the equivalence classes from subtask_2b, carefully accounting for the number of methyl or ethyl groups, their symmetry, and expected splitting patterns."
    N4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2b, answer2b], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, predicting aliphatic NMR signals, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction_5 = "Sub-task 5: Using a Self-Consistency Chain-of-Thought (SC CoT) approach, generate multiple independent predictions of the aromatic region NMR signals for each candidate compound and compare these predictions with the observed data (two singlets at ~6.7 ppm in a 1:1 ratio) to identify consistent matches."
    N5 = self.max_sc
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N5):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, predicting and comparing aromatic NMR signals, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    answer5_content = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[answer5_content]
    answer5 = answermapping_5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction_6a = "Sub-task 6a: Perform a detailed quantitative integration ratio analysis of the aliphatic region signals predicted in subtask_4, comparing them with the observed three singlets at ~2.2 ppm in a 2:1:1 ratio to evaluate compatibility."
    cot_instruction_6b = "Sub-task 6b: Analyze the splitting patterns and chemical shifts of the predicted aliphatic signals from subtask_4 and compare them with the observed singlets to assess the likelihood of each candidate compound's aliphatic environment matching the experimental data."
    N6 = self.max_sc
    cot_agents_6a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    cot_agents_6b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_answers_6a = []
    thinkingmapping_6a = {}
    answermapping_6a = {}
    possible_answers_6b = []
    thinkingmapping_6b = {}
    answermapping_6b = {}
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_instruction_6b,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N6):
        thinking6a, answer6a = await cot_agents_6a[i]([taskInfo, thinking4, answer4], cot_instruction_6a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6a[i].id}, analyzing integration ratios, thinking: {thinking6a.content}; answer: {answer6a.content}")
        possible_answers_6a.append(answer6a.content)
        thinkingmapping_6a[answer6a.content] = thinking6a
        answermapping_6a[answer6a.content] = answer6a
        thinking6b, answer6b = await cot_agents_6b[i]([taskInfo, thinking4, answer4], cot_instruction_6b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6b[i].id}, analyzing splitting patterns and chemical shifts, thinking: {thinking6b.content}; answer: {answer6b.content}")
        possible_answers_6b.append(answer6b.content)
        thinkingmapping_6b[answer6b.content] = thinking6b
        answermapping_6b[answer6b.content] = answer6b
    answer6a_content = Counter(possible_answers_6a).most_common(1)[0][0]
    thinking6a = thinkingmapping_6a[answer6a_content]
    answer6a = answermapping_6a[answer6a_content]
    answer6b_content = Counter(possible_answers_6b).most_common(1)[0][0]
    thinking6b = thinkingmapping_6b[answer6b_content]
    answer6b = answermapping_6b[answer6b_content]
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    debate_instruction_7 = "Sub-task 7: Conduct a Debate pattern analysis integrating the aromatic (subtask_5) and aliphatic (subtasks_6a and 6b) NMR data to explore conflicting interpretations and refine the identification of the two compounds present in the mixture, followed by a Reflexion step to critically evaluate assumptions about methyl equivalence and signal assignments."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6a", "answer of subtask 6a", "thinking of subtask 6b", "answer of subtask 6b"],
        "agent_collaboration": "Debate + Reflexion"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking5, answer5, thinking6a, answer6a, thinking6b, answer6b], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking5, answer5, thinking6a, answer6a, thinking6b, answer6b] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating integrated NMR data, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    cot_agent_reflexion = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_reflexion = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_reflexion = [taskInfo] + all_thinking7[-1] + all_answer7[-1]
    thinking7_reflect, answer7_reflect = await cot_agent_reflexion(cot_inputs_reflexion, "Sub-task 7 Reflexion: Critically evaluate the debate conclusions, reassess assumptions about methyl equivalence and signal assignments, and refine the identification of the compound pair.", 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, critically evaluating debate results, thinking: {thinking7_reflect.content}; answer: {answer7_reflect.content}")
    for i in range(self.max_round):
        feedback7, correct7 = await critic_agent_reflexion([taskInfo, thinking7_reflect, answer7_reflect], "Please review the reflexion analysis and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_reflexion.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_reflexion.extend([thinking7_reflect, answer7_reflect, feedback7])
        thinking7_reflect, answer7_reflect = await cot_agent_reflexion(cot_inputs_reflexion, "Sub-task 7 Reflexion: Refine the critical evaluation of debate conclusions.", i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, refining reflexion analysis, thinking: {thinking7_reflect.content}; answer: {answer7_reflect.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7_reflect.content}; answer - {answer7_reflect.content}")
    subtask_desc7['response'] = {"thinking": thinking7_reflect, "answer": answer7_reflect}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_instruction_8 = "Sub-task 8: Based on the integrated and critically evaluated NMR data from subtask_7, select the correct multiple-choice answer (A, B, C, or D) corresponding to the pair of compounds that best match the observed NMR spectral pattern in the 1:1 mixture."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7_reflect, answer7_reflect], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, selecting correct multiple-choice answer, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs