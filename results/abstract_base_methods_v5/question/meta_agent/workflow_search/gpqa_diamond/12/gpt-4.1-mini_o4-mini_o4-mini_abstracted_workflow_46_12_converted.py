async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Analyze the starting material (R)-(+)-Limonene with explicit atom numbering and stereochemical descriptors. "
        "Determine structural and stereochemical changes after partial hydrogenation with Pd/C under 1 equivalent hydrogen to identify product 1. "
        "Provide detailed stereochemical assignments and atom numbering in JSON format."
    )
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
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, analyzing starting material, thinking: {thinking1.content}; answer: {answer1.content}")
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

    cot_sc_instruction_2 = (
        "Sub-task 2: Determine the structural and stereochemical transformation of product 1 upon treatment with 3-chloroperbenzoic acid. "
        "Focus on epoxidation site, stereochemistry, and conformational analysis to identify product 2 with explicit stereochemical assignments and atom numbering in JSON format."
    )
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing epoxidation, thinking: {thinking2.content}; answer: {answer2.content}")
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

    cot_instruction_3a = (
        "Sub-task 3a: Perform detailed conformational analysis of the epoxide ring in product 2 within the cyclohexane chair conformation. "
        "Identify axial/equatorial positions and stereoelectronic factors influencing nucleophilic attack. Provide explicit atom numbering and stereochemical descriptors in JSON format."
    )
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, conformational analysis of epoxide ring, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_instruction_3b = (
        "Sub-task 3b: Evaluate regioselectivity and stereoelectronic effects governing nucleophilic ring-opening of the epoxide in product 2 by sodium methoxide. "
        "Identify site of attack (less substituted carbon), inversion of configuration, and possible stereoisomer mixtures. Provide detailed stereochemical reasoning and explicit atom numbering in JSON format."
    )
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, evaluating nucleophilic ring-opening, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_instruction_3c = (
        "Sub-task 3c: Predict the comprehensive stereochemical outcome of the epoxide ring-opening reaction. "
        "Assign absolute configurations (R/S) to all stereocenters in product 3, supported by CIP priority analysis and explicit atom numbering in JSON format."
    )
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking3c, answer3c = await cot_agent_3c([taskInfo, thinking3b, answer3b], cot_instruction_3c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3c.id}, stereochemical outcome prediction, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])

    cot_reflect_instruction_3d = (
        "Sub-task 3d: Conduct a reflexive review and debate of the stereochemical assignments in product 3. "
        "Reconcile alternative stereochemical pathways and ensure consensus or document minority opinions with valid reasoning. "
        "Use Reflexion and Debate patterns with explicit JSON outputs including atom numbering and stereochemical descriptors."
    )
    cot_agent_3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    debate_agents_3d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3d = self.max_round
    cot_inputs_3d = [taskInfo, thinking3c, answer3c]
    subtask_desc3d = {
        "subtask_id": "subtask_3d",
        "instruction": cot_reflect_instruction_3d,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Reflexion_Debate"
    }
    thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, initial stereochemical review, thinking: {thinking3d.content}; answer: {answer3d.content}")
    for i in range(N_max_3d):
        feedback, correct = await critic_agent_3d([taskInfo, thinking3d, answer3d], "Review stereochemical assignments and provide limitations or alternative valid assignments.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3d.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3d.extend([thinking3d, answer3d, feedback])
        thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, refining stereochemical assignments, thinking: {thinking3d.content}; answer: {answer3d.content}")
    all_thinking_3d = [[] for _ in range(N_max_3d)]
    all_answer_3d = [[] for _ in range(N_max_3d)]
    for r in range(N_max_3d):
        for i, agent in enumerate(debate_agents_3d):
            if r == 0:
                thinking_tmp, answer_tmp = await agent([taskInfo, thinking3d, answer3d], cot_reflect_instruction_3d, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3d, answer3d] + all_thinking_3d[r-1] + all_answer_3d[r-1]
                thinking_tmp, answer_tmp = await agent(input_infos, cot_reflect_instruction_3d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating stereochemical assignments, thinking: {thinking_tmp.content}; answer: {answer_tmp.content}")
            all_thinking_3d[r].append(thinking_tmp)
            all_answer_3d[r].append(answer_tmp)
    final_decision_agent_3d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3d_final, answer3d_final = await final_decision_agent_3d([taskInfo] + all_thinking_3d[-1] + all_answer_3d[-1], "Sub-task 3d: Make final decision on stereochemical assignments of product 3 considering minority opinions.", is_sub_task=True)
    agents.append(f"Final Decision agent, stereochemical arbitration, thinking: {thinking3d_final.content}; answer: {answer3d_final.content}")
    sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d_final.content}; answer - {answer3d_final.content}")
    subtask_desc3d['response'] = {"thinking": thinking3d_final, "answer": answer3d_final}
    logs.append(subtask_desc3d)
    print("Step 3d: ", sub_tasks[-1])

    cot_instruction_4a = (
        "Sub-task 4a: Analyze the esterification reaction of product 3 with propanoic acid, DCC, and catalytic DMAP. "
        "Focus on conformational determination of substituent orientations (axial/equatorial) in product 4 and their influence on stereochemistry. "
        "Provide explicit atom numbering and stereochemical descriptors in JSON format."
    )
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 3d", "answer of subtask 3d"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3d_final, answer3d_final], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, conformational analysis of esterification, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    cot_instruction_4b = (
        "Sub-task 4b: Perform rigorous CIP priority assignments for each stereocenter in product 4. "
        "Determine absolute configurations (R/S) and consider possible isomer mixtures with explicit atom numbering and stereochemical descriptors in JSON format."
    )
    cip_checker_agent = LLMAgentBase(["thinking", "answer"], "CIP Checker Agent", model=self.node_model, temperature=0.0)
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "CoT"
    }
    thinking4b, answer4b = await cip_checker_agent([taskInfo, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
    agents.append(f"CIP Checker agent {cip_checker_agent.id}, assigning CIP priorities, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    debate_instruction_4c = (
        "Sub-task 4c: Conduct a debate and arbitration process to resolve any stereochemical ambiguities or conflicting assignments in product 4. "
        "Ensure minority opinions are considered and final stereochemical assignments are robust and well-justified. "
        "Use Debate pattern with explicit JSON outputs including atom numbering and stereochemical descriptors."
    )
    debate_agents_4c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4c = self.max_round
    all_thinking_4c = [[] for _ in range(N_max_4c)]
    all_answer_4c = [[] for _ in range(N_max_4c)]
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": debate_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4c):
        for i, agent in enumerate(debate_agents_4c):
            if r == 0:
                thinking4c, answer4c = await agent([taskInfo, thinking4b, answer4b], debate_instruction_4c, r, is_sub_task=True)
            else:
                input_infos_4c = [taskInfo, thinking4b, answer4b] + all_thinking_4c[r-1] + all_answer_4c[r-1]
                thinking4c, answer4c = await agent(input_infos_4c, debate_instruction_4c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating stereochemical assignments of product 4, thinking: {thinking4c.content}; answer: {answer4c.content}")
            all_thinking_4c[r].append(thinking4c)
            all_answer_4c[r].append(answer4c)
    final_decision_agent_4c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4c_final, answer4c_final = await final_decision_agent_4c([taskInfo] + all_thinking_4c[-1] + all_answer_4c[-1], "Sub-task 4c: Make final decision on stereochemical assignments of product 4 considering minority opinions.", is_sub_task=True)
    agents.append(f"Final Decision agent, stereochemical arbitration of product 4, thinking: {thinking4c_final.content}; answer: {answer4c_final.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c_final.content}; answer - {answer4c_final.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c_final, "answer": answer4c_final}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Compare the finalized stereochemical structure(s) of product 4 with the given multiple-choice options. "
        "Analyze stereochemistry, substituents, and functional groups to select the valid structure that matches the entire reaction sequence and conditions. "
        "Provide detailed justification and explicit mapping to the choices in JSON format."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4c", "answer of subtask 4c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4c_final, answer4c_final], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4c_final, answer4c_final] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting valid product 4 structure, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_5[r].append(thinking5)
            all_answer_5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5_final, answer5_final = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Make final decision on valid product 4 structure.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting product 4 structure, thinking: {thinking5_final.content}; answer: {answer5_final.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5_final.content}; answer - {answer5_final.content}")
    subtask_desc5['response'] = {"thinking": thinking5_final, "answer": answer5_final}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Perform a final sanity check and validation of the selected product 4 structure. "
        "Verify stereochemical assignments, conformations, and consistency with reaction mechanisms to ensure correctness before final answer submission. "
        "Provide detailed validation report in JSON format."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5_final, answer5_final], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, final sanity check, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
