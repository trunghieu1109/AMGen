async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agents_sc = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]

    cot_reflect_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(3)]
    critic_agents = [LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0) for _ in range(3)]

    # Stage 1: Analyze the components of reactions A, B, and C with Self-Consistency CoT considering stereochemistry, tautomerism, protonation

    cot_instruction_1 = "Sub-task 1: Analyze the components of reaction A (dimethyl malonate + methyl (E)-3-(p-tolyl)acrylate + NaOEt, EtOH). Identify the nucleophile, electrophile, and reaction conditions. Predict the site of nucleophilic attack, intermediate formation, and consider resonance, stereochemistry, and tautomerism relevant to the Michael addition."
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_agents_sc[i]([taskInfo], cot_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, analyzing reaction A, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    answer1_content = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[answer1_content]
    answer1 = answermapping_1[answer1_content]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = "Sub-task 2: Analyze the components of reaction B (1-(cyclohex-1-en-1-yl)piperidine + (E)-but-2-enenitrile + MeOH, H3O+). Identify the nucleophile, electrophile, and reaction conditions. Predict the site of nucleophilic attack, intermediate formation, and consider resonance, stereochemistry, tautomerism, and protonation states relevant to the Michael addition."
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents_sc[i]([taskInfo], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, analyzing reaction B, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = "Sub-task 3: Analyze the components of reaction C (unknown reactant C + but-3-en-2-one + KOH, H2O). Identify the nucleophile, electrophile, and reaction conditions. Deduce the identity of reactant C based on the product name 2-(3-oxobutyl)cyclohexane-1,3-dione. Predict the site of nucleophilic attack, intermediate formation, and consider resonance, stereochemistry, tautomerism, and protonation states relevant to the Michael addition."
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_agents_sc[i]([taskInfo], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, analyzing reaction C, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Deduce detailed major final products for reactions A, B, and C with Reflexion considering stereochemistry, tautomerism, protonation

    cot_reflect_instruction_4 = "Sub-task 4: For reaction A, based on the analysis in subtask_1, deduce the detailed major final product structure after the Michael addition and subsequent steps, explicitly analyzing stereochemistry, tautomerism, and protonation states. Generate possible tautomeric or resonance forms and select the most chemically consistent product."
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_4 = [taskInfo, thinking1, answer1]
    cot_agent_4 = cot_reflect_agents[0]
    critic_agent_4 = critic_agents[0]
    N_max_4 = self.max_round
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, deducing product for reaction A, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the product deduction for reaction A and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining product deduction for reaction A, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_reflect_instruction_5 = "Sub-task 5: For reaction B, based on the analysis in subtask_2, deduce the detailed major final product structure after the Michael addition and subsequent steps, explicitly analyzing stereochemistry, tautomerism, and protonation states. Generate possible tautomeric or resonance forms and select the most chemically consistent product."
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_5 = [taskInfo, thinking2, answer2]
    cot_agent_5 = cot_reflect_agents[1]
    critic_agent_5 = critic_agents[1]
    N_max_5 = self.max_round
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, deducing product for reaction B, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please review the product deduction for reaction B and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining product deduction for reaction B, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction_6 = "Sub-task 6: For reaction C, based on the analysis in subtask_3, deduce the detailed identity of reactant C and the major final product structure after the Michael addition and subsequent steps, explicitly analyzing stereochemistry, tautomerism, and protonation states. Generate possible tautomeric or resonance forms and select the most chemically consistent product matching the given product name."
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_6 = [taskInfo, thinking3, answer3]
    cot_agent_6 = cot_reflect_agents[2]
    critic_agent_6 = critic_agents[2]
    N_max_6 = self.max_round
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, deducing product and reactant C for reaction C, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please review the product and reactant C deduction for reaction C and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining product and reactant C deduction for reaction C, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Stage 3: Exact string-for-string comparison between deduced product names and multiple-choice options

    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": "Sub-task 7: Perform an exact string-for-string comparison between the deduced product names (including stereochemistry and tautomeric forms) for reactions A, B, and C and each multiple-choice option. Identify any mismatches or ambiguities and flag them for re-examination if no exact match is found.",
        "context": ["user query", "thinking and answer of subtask 4", "thinking and answer of subtask 5", "thinking and answer of subtask 6"],
        "agent_collaboration": "Debate"
    }

    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6], subtask_desc7["instruction"], r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, subtask_desc7["instruction"], r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing deduced products with choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    subtask_desc7['response'] = {"thinking": all_thinking7[-1], "answer": all_answer7[-1]}
    logs.append(subtask_desc7)
    print("Step 7: ", "Completed exact matching and ambiguity check")

    # Subtask 8: Final decision based on validation in subtask 7

    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 8: Based on the validation in subtask_7, select the correct multiple-choice answer (A, B, C, or D) that exactly matches the deduced products and reactants for all three reactions. If no exact match exists, report the ambiguity and recommend re-evaluation.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct multiple-choice answer, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": "Sub-task 8: Final answer selection based on exact matching and ambiguity check.",
        "context": ["user query", "outputs of subtask 7"],
        "agent_collaboration": "Final Decision"
    }
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
