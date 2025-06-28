async def forward_52(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the FTIR spectrum data to confirm the presence and specific absorption peaks of the ester functional group, noting exact peak positions and intensities to characterize the ester environment."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyze FTIR spectrum for ester group, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Analyze the 1H NMR spectrum data in detail: identify and categorize the six proton signals by chemical shift, integration, multiplicity (including coupling constants), and assign them to aromatic-H, vinyl-H (with specified multiplicities: one doublet and one doublet of quartets), and methyl (-CH3) groups, explicitly noting the absence of -CH2 signals, using self-consistency to ensure accuracy."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyze 1H NMR signals in detail, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_sc_instruction_2_5 = "Sub-task 2.5: Map each identified 1H NMR signal to corresponding structural fragments, explicitly listing the number of carbons, hydrogens, and oxygens in each fragment, to build a detailed atom count per substituent."
    cot_agents_2_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2_5 = []
    thinkingmapping_2_5 = {}
    answermapping_2_5 = {}
    subtask_desc2_5 = {
        "subtask_id": "subtask_2.5",
        "instruction": cot_sc_instruction_2_5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2_5, answer2_5 = await cot_agents_2_5[i]([taskInfo, thinking2, answer2], cot_sc_instruction_2_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_5[i].id}, map NMR signals to structural fragments with atom counts, thinking: {thinking2_5.content}; answer: {answer2_5.content}")
        possible_answers_2_5.append(answer2_5.content)
        thinkingmapping_2_5[answer2_5.content] = thinking2_5
        answermapping_2_5[answer2_5.content] = answer2_5
    answer2_5_content = Counter(possible_answers_2_5).most_common(1)[0][0]
    thinking2_5 = thinkingmapping_2_5[answer2_5_content]
    answer2_5 = answermapping_2_5[answer2_5_content]
    sub_tasks.append(f"Sub-task 2.5 output: thinking - {thinking2_5.content}; answer - {answer2_5.content}")
    subtask_desc2_5['response'] = {
        "thinking": thinking2_5,
        "answer": answer2_5
    }
    logs.append(subtask_desc2_5)
    print("Step 2.5: ", sub_tasks[-1])
    
    cot_reflect_instruction_3a = "Sub-task 3a: Analyze the aromatic substitution pattern on the 6-membered ring using the aromatic-H signals and FTIR ester data to deduce possible substitution sites and their impact on hydrogen count and chemical environment."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3a = self.max_round
    cot_inputs_3a = [taskInfo, thinking1, answer1, thinking2_5, answer2_5]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2.5"],
        "agent_collaboration": "Reflexion"
    }
    thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, analyze aromatic substitution pattern, thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_max_3a):
        feedback, correct = await critic_agent_3a([taskInfo, thinking3a, answer3a], "please review the aromatic substitution pattern analysis and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback])
        thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining aromatic substitution pattern, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_reflect_instruction_3b = "Sub-task 3b: Evaluate the vinyl-H signals (doublet and doublet of quartets) to determine the vinyl group structure, coupling relationships, and corresponding hydrogen and carbon counts."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking2_5, answer2_5]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking and answer of subtask 2.5"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, evaluate vinyl-H signals, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "please review the vinyl group structure and coupling analysis and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining vinyl group analysis, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_reflect_instruction_3c = "Sub-task 3c: Assign the methyl (-CH3) groups based on their NMR signals and integrate their atom counts into the overall molecular structure."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3c = self.max_round
    cot_inputs_3c = [taskInfo, thinking2_5, answer2_5]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking and answer of subtask 2.5"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, assign methyl groups and atom counts, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max_3c):
        feedback, correct = await critic_agent_3c([taskInfo, thinking3c, answer3c], "please review the methyl group assignment and atom count integration and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining methyl group assignment, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    
    cot_reflect_instruction_4 = "Sub-task 4: Aggregate atom counts (C, H, O) from all assigned substituents and the aromatic ring, calculate total hydrogens and carbons, and perform degrees of unsaturation analysis; cross-validate these totals against each candidate molecular formula (C11H12O2, C11H14O2, C12H12O2, C12H14O2) to identify compatible formulas."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking and answer of subtasks 3a, 3b, 3c"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, aggregate atom counts and cross-validate molecular formulas, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the atom count aggregation, degrees of unsaturation, and molecular formula cross-validation for correctness and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining atom count aggregation and formula validation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_reflect_instruction_4_5 = "Sub-task 4.5: Conduct a reflexion checkpoint to critically reassess atom counts, spectral data consistency, and substitution patterns; explicitly justify exclusion or inclusion of each molecular formula candidate based on spectral and structural evidence."
    cot_agent_4_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4_5 = self.max_round
    cot_inputs_4_5 = [taskInfo, thinking4, answer4]
    subtask_desc4_5 = {
        "subtask_id": "subtask_4.5",
        "instruction": cot_reflect_instruction_4_5,
        "context": ["user query", "thinking and answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking4_5, answer4_5 = await cot_agent_4_5(cot_inputs_4_5, cot_reflect_instruction_4_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_5.id}, reflexion checkpoint on atom counts and formula justification, thinking: {thinking4_5.content}; answer: {answer4_5.content}")
    for i in range(N_max_4_5):
        feedback, correct = await critic_agent_4_5([taskInfo, thinking4_5, answer4_5], "please critically reassess the atom counts, spectral consistency, and molecular formula justification.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4_5.extend([thinking4_5, answer4_5, feedback])
        thinking4_5, answer4_5 = await cot_agent_4_5(cot_inputs_4_5, cot_reflect_instruction_4_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_5.id}, refining reflexion checkpoint, thinking: {thinking4_5.content}; answer: {answer4_5.content}")
    sub_tasks.append(f"Sub-task 4.5 output: thinking - {thinking4_5.content}; answer - {answer4_5.content}")
    subtask_desc4_5['response'] = {
        "thinking": thinking4_5,
        "answer": answer4_5
    }
    logs.append(subtask_desc4_5)
    print("Step 4.5: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Select the correct molecular formula from the given choices (A: C12H14O2, B: C12H12O2, C: C11H14O2, D: C11H12O2) based on comprehensive spectral analysis, atom counting, and reflexion; provide the corresponding letter choice as the final answer."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4.5", "answer of subtask 4.5"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4_5, answer4_5], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4_5, answer4_5] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting molecular formula, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct molecular formula and provide the letter choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final molecular formula selection, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
