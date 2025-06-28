async def forward_130(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_1 = "Sub-task 1: Extract and tabulate all 1H NMR signals of the major product, including chemical shifts, multiplicities, integrals, and assign candidate proton types (e.g., methyl, methylene, methine) based on typical chemical shift ranges and integrals."
    N1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, extracting and assigning 1H NMR signals, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    most_common_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[most_common_answer_1]
    answer1 = answermapping_1[most_common_answer_1]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Extract and tabulate all relevant 13C NMR signals of the anhydride and major product, correlating them with possible carbon environments to support proton assignments, based on output from Sub-task 1."
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, extracting and assigning 13C NMR signals, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_reflect_instruction_3 = "Sub-task 3: Describe and illustrate the 3D geometry of the bicyclic adducts (major and minor products), including endo and exo stereochemical possibilities, and estimate approximate interproton distances relevant to NOESY correlations, based on outputs from Sub-task 2."
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
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, describing 3D geometry and stereochemistry, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the 3D geometry description and interproton distance estimations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining 3D geometry and stereochemistry, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instruction_4 = "Sub-task 4: Critically analyze the 2D NOESY spectra of the major and minor products, identifying observed cross-peaks and their absence, and link these to spatial proximity of specific protons based on the 3D geometry and proton assignments from previous subtasks."
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
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
                input_infos_4.extend(all_answer4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing NOESY spectra, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make a final decision on NOESY cross-peak analysis.", is_sub_task=True)
    agents.append(f"Final Decision agent on NOESY analysis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_reflect_instruction_5a = "Sub-task 5a: Assign all 1H NMR signals of the major product to specific protons in the bicyclic structure, using chemical shifts, integrals, multiplicities, and structural context, based on outputs from Sub-task 4."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5a = self.max_round
    cot_inputs_5a = [taskInfo, thinking4, answer4]
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_reflect_instruction_5a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5a, answer5a = await cot_agent_5a(cot_inputs_5a, cot_reflect_instruction_5a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5a.id}, assigning 1H NMR signals to protons, thinking: {thinking5a.content}; answer: {answer5a.content}")
    for i in range(N_max_5a):
        feedback, correct = await critic_agent_5a([taskInfo, thinking5a, answer5a], "please review the proton assignments for accuracy and consistency.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5a.extend([thinking5a, answer5a, feedback])
        thinking5a, answer5a = await cot_agent_5a(cot_inputs_5a, cot_reflect_instruction_5a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5a.id}, refining proton assignments, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_reflect_instruction_5b = "Sub-task 5b: Predict expected NOESY correlations for the major product based on spatial proximity of assigned protons, considering stereochemistry and interproton distances, based on outputs from Sub-task 5a."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5b = self.max_round
    cot_inputs_5b = [taskInfo, thinking5a, answer5a]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_reflect_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "Reflexion"
    }
    thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_reflect_instruction_5b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, predicting NOESY correlations, thinking: {thinking5b.content}; answer: {answer5b.content}")
    for i in range(N_max_5b):
        feedback, correct = await critic_agent_5b([taskInfo, thinking5b, answer5b], "please review the predicted NOESY correlations for consistency with spatial proximity.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5b.extend([thinking5b, answer5b, feedback])
        thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_reflect_instruction_5b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, refining NOESY correlation predictions, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    debate_instruction_5c = "Sub-task 5c: Match the predicted NOESY correlations to the observed cross-peak in the major product’s NOESY spectrum, identifying which two resonances correspond to the cross-peak and justifying the assignment with chemical and spatial evidence, based on outputs from Sub-task 5b."
    debate_agents_5c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5c = self.max_round
    all_thinking5c = [[] for _ in range(N_max_5c)]
    all_answer5c = [[] for _ in range(N_max_5c)]
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": debate_instruction_5c,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5c):
        for i, agent in enumerate(debate_agents_5c):
            if r == 0:
                thinking5c, answer5c = await agent([taskInfo, thinking5b, answer5b], debate_instruction_5c, r, is_sub_task=True)
            else:
                input_infos_5c = [taskInfo, thinking5b, answer5b] + all_thinking5c[r-1] + all_answer5c[r-1]
                thinking5c, answer5c = await agent(input_infos_5c, debate_instruction_5c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, matching predicted to observed NOESY cross-peaks, thinking: {thinking5c.content}; answer: {answer5c.content}")
            all_thinking5c[r].append(thinking5c)
            all_answer5c[r].append(answer5c)
    final_decision_agent_5c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5c, answer5c = await final_decision_agent_5c([taskInfo] + all_thinking5c[-1] + all_answer5c[-1], "Sub-task 5c: Make final decision on which two resonances correspond to the NOESY cross-peak.", is_sub_task=True)
    agents.append(f"Final Decision agent on NOESY cross-peak assignment, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {"thinking": thinking5c, "answer": answer5c}
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    cot_reflect_instruction_6 = "Sub-task 6: Perform a reflexive evaluation of the NOESY cross-peak assignment by re-examining alternative proton assignments and stereochemical possibilities to confirm or revise the final conclusion, based on outputs from Sub-task 5c."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5c, answer5c]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, reflexive evaluation of NOESY assignment, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please critically evaluate the NOESY cross-peak assignment and consider alternative assignments.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining reflexive evaluation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs