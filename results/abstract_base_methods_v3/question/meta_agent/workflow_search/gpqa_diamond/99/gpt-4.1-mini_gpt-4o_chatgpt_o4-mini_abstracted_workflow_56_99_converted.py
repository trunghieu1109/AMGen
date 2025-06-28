async def forward_99(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify the structure and type of compound A (C3H6) and determine the product B formed after bromination in carbon tetrachloride, including the reaction mechanism and structural features of B."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying compound A and product B, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Determine the structure and properties of compound C formed when compound B reacts with alcoholic KOH, including the type of reaction and physical state of C, based on Sub-task 1 outputs."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining compound C, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Sub-task 3: Identify compound D formed by passing compound C through a red-hot iron tube, analyze its structure, and predict its 1H NMR spectral characteristics relevant to the given statement, based on Sub-task 2 outputs."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, identifying compound D and analyzing NMR features, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the identification and NMR analysis of compound D and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining compound D identification and NMR analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instruction_4 = "Sub-task 4: Determine the structure and characteristics of compound E formed when compound D reacts with a mixture of two strong acids, including the nature of the acids and the reaction type, based on Sub-task 3 outputs."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining compound E, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on compound E structure and characteristics.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding compound E, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5 = "Sub-task 5: Identify compound F formed when compound E reacts with iron scrap and hydrochloric acid, analyze its structure, and evaluate its common uses, especially in dye synthesis, based on Sub-task 4 outputs."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, identifying compound F and evaluating uses, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_sc_instruction_6 = "Sub-task 6: Determine the structure and properties of compound G formed when compound F reacts with nitrous acid, including the reaction mechanism and functional groups involved, based on Sub-task 5 outputs."
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, determining compound G, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    most_common_answer_6 = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[most_common_answer_6]
    answer6 = answermapping_6[most_common_answer_6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_reflect_instruction_7 = "Sub-task 7: Identify compound H formed when compound G reacts with sodium hydroxide, analyze its chemical behavior, and assess its reaction with ferric chloride solution, including color changes, based on Sub-task 6 outputs."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, identifying compound H and analyzing chemical behavior, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7], "please review the identification and chemical behavior analysis of compound H and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining compound H identification and chemical behavior analysis, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Sub-task 8: Analyze the 1H NMR spectral characteristics of compound D to verify the claim that it gives two singlets, based on Sub-task 3 outputs."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking3, answer3], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, analyzing NMR spectra of compound D, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    cot_instruction_9 = "Sub-task 9: Evaluate the claim that compound F is used for the synthesis of dyes by reviewing its chemical identity and known industrial applications, based on Sub-task 5 outputs."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking5, answer5], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, evaluating dye synthesis claim of compound F, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    cot_instruction_10 = "Sub-task 10: Assess whether compound H gives a yellow color with ferric chloride solution, indicating the presence or absence of phenolic or related groups, based on Sub-task 7 outputs."
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction_10,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking7, answer7], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, assessing color reaction of compound H, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    
    cot_instruction_11 = "Sub-task 11: Verify if compound C is a flammable gas by analyzing its molecular structure, physical state, and known properties, based on Sub-task 2 outputs."
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_instruction_11,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking11, answer11 = await cot_agent_11([taskInfo, thinking2, answer2], cot_instruction_11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_11.id}, verifying flammability of compound C, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])
    
    cot_sc_instruction_12a = "Sub-task 12a: Identify all incorrect statements among the given choices by comparing the results from Sub-tasks 8, 9, 10, and 11, explicitly listing each incorrect statement with justification."
    cot_agents_12a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_12a = []
    thinkingmapping_12a = {}
    answermapping_12a = {}
    subtask_desc12a = {
        "subtask_id": "subtask_12a",
        "instruction": cot_sc_instruction_12a,
        "context": ["user query", "thinking and answers of subtasks 8,9,10,11", "answers of subtasks 8,9,10,11"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking12a, answer12a = await cot_agents_12a[i]([taskInfo, thinking8, answer8, thinking9, answer9, thinking10, answer10, thinking11, answer11], cot_sc_instruction_12a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_12a[i].id}, identifying all incorrect statements, thinking: {thinking12a.content}; answer: {answer12a.content}")
        possible_answers_12a.append(answer12a.content)
        thinkingmapping_12a[answer12a.content] = thinking12a
        answermapping_12a[answer12a.content] = answer12a
    most_common_answer_12a = Counter(possible_answers_12a).most_common(1)[0][0]
    thinking12a = thinkingmapping_12a[most_common_answer_12a]
    answer12a = answermapping_12a[most_common_answer_12a]
    sub_tasks.append(f"Sub-task 12a output: thinking - {thinking12a.content}; answer - {answer12a.content}")
    subtask_desc12a['response'] = {"thinking": thinking12a, "answer": answer12a}
    logs.append(subtask_desc12a)
    print("Step 12a: ", sub_tasks[-1])
    
    debate_instruction_12b = "Sub-task 12b: Prioritize or select the primary incorrect statement from the list identified in Sub-task 12a, providing clear reasoning for the choice in alignment with the question's intent and ambiguity resolution."
    debate_agents_12b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_12b = self.max_round
    all_thinking12b = [[] for _ in range(N_max_12b)]
    all_answer12b = [[] for _ in range(N_max_12b)]
    subtask_desc12b = {
        "subtask_id": "subtask_12b",
        "instruction": debate_instruction_12b,
        "context": ["user query", "thinking and answer of subtask 12a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_12b):
        for i, agent in enumerate(debate_agents_12b):
            if r == 0:
                thinking12b, answer12b = await agent([taskInfo, thinking12a, answer12a], debate_instruction_12b, r, is_sub_task=True)
            else:
                input_infos_12b = [taskInfo, thinking12a, answer12a] + all_thinking12b[r-1] + all_answer12b[r-1]
                thinking12b, answer12b = await agent(input_infos_12b, debate_instruction_12b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, prioritizing primary incorrect statement, thinking: {thinking12b.content}; answer: {answer12b.content}")
            all_thinking12b[r].append(thinking12b)
            all_answer12b[r].append(answer12b)
    final_decision_agent_12b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking12b, answer12b = await final_decision_agent_12b([taskInfo] + all_thinking12b[-1] + all_answer12b[-1], "Sub-task 12b: Make final decision on the primary incorrect statement.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding primary incorrect statement, thinking: {thinking12b.content}; answer: {answer12b.content}")
    sub_tasks.append(f"Sub-task 12b output: thinking - {thinking12b.content}; answer - {answer12b.content}")
    subtask_desc12b['response'] = {"thinking": thinking12b, "answer": answer12b}
    logs.append(subtask_desc12b)
    print("Step 12b: ", sub_tasks[-1])
    
    cot_reflect_instruction_13 = "Sub-task 13: Perform a final validation and review of the identified incorrect statement(s) and the prioritized answer to ensure consistency, clarity, and alignment with the question's requirements before final output."
    cot_agent_13 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_13 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_13 = self.max_round
    cot_inputs_13 = [taskInfo, thinking12a, answer12a, thinking12b, answer12b]
    subtask_desc13 = {
        "subtask_id": "subtask_13",
        "instruction": cot_reflect_instruction_13,
        "context": ["user query", "thinking and answers of subtasks 12a and 12b"],
        "agent_collaboration": "Reflexion"
    }
    thinking13, answer13 = await cot_agent_13(cot_inputs_13, cot_reflect_instruction_13, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_13.id}, performing final validation and review, thinking: {thinking13.content}; answer: {answer13.content}")
    for i in range(N_max_13):
        feedback, correct = await critic_agent_13([taskInfo, thinking13, answer13], "please review the final validation and confirm alignment with question requirements.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_13.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_13.extend([thinking13, answer13, feedback])
        thinking13, answer13 = await cot_agent_13(cot_inputs_13, cot_reflect_instruction_13, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_13.id}, refining final validation, thinking: {thinking13.content}; answer: {answer13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking13.content}; answer - {answer13.content}")
    subtask_desc13['response'] = {"thinking": thinking13, "answer": answer13}
    logs.append(subtask_desc13)
    print("Step 13: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking13, answer13, sub_tasks, agents)
    return final_answer, logs
