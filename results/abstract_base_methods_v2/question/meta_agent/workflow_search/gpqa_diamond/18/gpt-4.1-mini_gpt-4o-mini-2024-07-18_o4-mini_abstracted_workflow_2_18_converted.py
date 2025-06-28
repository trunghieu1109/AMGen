async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agents_sc = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    critic_agent_mid = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_agent_reflexion = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)

    cot_instruction_1a = (
        "Sub-task 1a: Explicitly identify and verify the nucleophile and electrophile in the reaction between methyl 2-oxocyclohexane-1-carboxylate, NaOEt, and 2,4-dimethyl-1-(vinylsulfinyl)benzene. "
        "Analyze their structures including SMILES or structural diagrams, determine which species forms the enolate and which acts as the Michael acceptor, and clarify reactive sites with correct IUPAC numbering."
    )
    cot_instruction_1b = (
        "Sub-task 1b: Confirm the nucleophilic attack site and predict the key intermediates formed in the reaction from Sub-task 1a, including stereochemical considerations such as chiral centers and possible stereoisomers."
    )

    cot_instruction_2a = (
        "Sub-task 2a: Explicitly identify and verify the nucleophile and electrophile in the reaction between ethyl 2-ethylbutanoate, NaH, and methyl 2-cyclopentylidene-2-phenylacetate. "
        "Analyze their structures including SMILES or structural diagrams, determine which species forms the enolate and which acts as the Michael acceptor, and clarify reactive sites with correct IUPAC numbering."
    )
    cot_instruction_2b = (
        "Sub-task 2b: Confirm the nucleophilic attack site and predict the key intermediates formed in the reaction from Sub-task 2a, including stereochemical considerations such as chiral centers and possible stereoisomers."
    )

    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_sc[i]([taskInfo], cot_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, Subtask 1a nucleophile/electrophile identification, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1a.append(answer.content)
        thinkingmapping_1a[answer.content] = thinking
        answermapping_1a[answer.content] = answer
    answer1a_content = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[answer1a_content]
    answer1a = answermapping_1a[answer1a_content]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])

    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_sc[i]([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, Subtask 1b nucleophilic attack site and intermediate prediction, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1b.append(answer.content)
        thinkingmapping_1b[answer.content] = thinking
        answermapping_1b[answer.content] = answer
    answer1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[answer1b_content]
    answer1b = answermapping_1b[answer1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])

    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_sc[i]([taskInfo], cot_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, Subtask 2a nucleophile/electrophile identification, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2a.append(answer.content)
        thinkingmapping_2a[answer.content] = thinking
        answermapping_2a[answer.content] = answer
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])

    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_sc[i]([taskInfo, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, Subtask 2b nucleophilic attack site and intermediate prediction, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2b.append(answer.content)
        thinkingmapping_2b[answer.content] = thinking
        answermapping_2b[answer.content] = answer
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])

    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": "Sub-task 3: Generate multiple mechanistic hypotheses for the Michael addition in both reactions (from subtasks 1b and 2b) using a Self-Consistency Chain-of-Thought approach, including reversed nucleophile/electrophile roles, and select the most chemically consistent pathways for each reaction.",
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_sc[i]([taskInfo, thinking1b, answer1b, thinking2b, answer2b], subtask_desc_3["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, Subtask 3 mechanistic hypotheses generation, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_3.append(answer.content)
        thinkingmapping_3[answer.content] = thinking
        answermapping_3[answer.content] = answer
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": "Sub-task 4: Perform a mid-pipeline critical validation of the nucleophile/electrophile assignments and mechanistic pathways identified in subtask_3, checking for consistency with known Michael reaction principles and correcting any fundamental errors before proceeding.",
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    N_max_4 = self.max_round
    thinking4, answer4 = await cot_agent_reflexion(cot_inputs_4, subtask_desc_4["instruction"], 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, mid-pipeline critical validation, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_mid([taskInfo, thinking4, answer4], "Please review the nucleophile/electrophile assignments and mechanistic pathways for correctness and consistency.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_mid.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_reflexion(cot_inputs_4, subtask_desc_4["instruction"], i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, refining mid-pipeline validation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Predict the intermediate and final product structures for reaction A by mapping new carbon-carbon bond formation onto the skeletal structure, "
        "assigning correct IUPAC numbering, and explicitly analyzing stereochemical outcomes including chiral centers and stereoisomers based on the validated mechanism from subtask_4."
    )
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    N_max_5 = self.max_round
    thinking5, answer5 = await cot_agent_reflexion(cot_inputs_5, cot_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, predicting product A structure, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_mid([taskInfo, thinking5, answer5], "Please review the predicted product A structure for stereochemical correctness and nomenclature accuracy.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_mid.id}, providing feedback on product A, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_reflexion(cot_inputs_5, cot_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, refining product A structure, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Predict the intermediate and final product structures for reaction B by mapping new carbon-carbon bond formation onto the skeletal structure, "
        "assigning correct IUPAC numbering, and explicitly analyzing stereochemical outcomes including chiral centers and stereoisomers based on the validated mechanism from subtask_4."
    )
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_6 = [taskInfo, thinking4, answer4]
    N_max_6 = self.max_round
    thinking6, answer6 = await cot_agent_reflexion(cot_inputs_6, cot_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, predicting product B structure, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_mid([taskInfo, thinking6, answer6], "Please review the predicted product B structure for stereochemical correctness and nomenclature accuracy.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_mid.id}, providing feedback on product B, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_reflexion(cot_inputs_6, cot_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, refining product B structure, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc_6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    debate_instruction_7 = (
        "Sub-task 7: Conduct a debate-style comparative analysis of the predicted products A and B (from subtasks 5 and 6) against the given multiple-choice options, "
        "focusing on mechanistic consistency, stereochemical correctness, and nomenclature accuracy to identify the correct product assignments."
    )
    subtask_desc_7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask_5", "answer of subtask_5", "thinking of subtask_6", "answer of subtask_6"],
        "agent_collaboration": "Debate"
    }
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking5, answer5, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking5, answer5, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating product assignments, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    thinking7_final, answer7_final = await final_decision_agent([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct product assignments based on debate.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on product assignments, thinking: {thinking7_final.content}; answer: {answer7_final.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7_final.content}; answer - {answer7_final.content}")
    subtask_desc_7['response'] = {"thinking": thinking7_final, "answer": answer7_final}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])

    subtask_desc_8 = {
        "subtask_id": "subtask_8",
        "instruction": "Sub-task 8: Perform a final validation and cross-check of the entire mechanistic pathway, stereochemical assignments, and product structures against established chemical principles and literature precedents to confirm the correctness of the selected multiple-choice answer.",
        "context": ["user query", "thinking of subtask_7", "answer of subtask_7"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_8 = [taskInfo, thinking7_final, answer7_final]
    N_max_8 = self.max_round
    thinking8, answer8 = await cot_agent_reflexion(cot_inputs_8, subtask_desc_8["instruction"], 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, final validation and cross-check, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback, correct = await critic_agent_mid([taskInfo, thinking8, answer8], "Please critically validate the entire mechanistic pathway and product assignments for correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_mid.id}, providing feedback on final validation, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_reflexion(cot_inputs_8, subtask_desc_8["instruction"], i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, refining final validation, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc_8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc_8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
