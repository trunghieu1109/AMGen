async def forward_121(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Determine the chemical structure of product 1 formed by treating acetic acid with bromine, pyridine, and acetic anhydride under heating, including identification of functional groups, substitution pattern, and provide a clear structural representation (e.g., Lewis structure or SMILES)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, determining product 1 structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Determine the chemical structure of product 2 formed by heating product 1 with ethanol and a small amount of sulfuric acid, including any esterification or substitution changes, and provide a clear structural representation."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining product 2 structure, thinking: {thinking2.content}; answer: {answer2.content}")
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

    cot_reflect_instruction_3 = "Sub-task 3: Determine the chemical structure of product 3 formed by treating product 2 with sodium cyanide, identifying the site and nature of cyanide substitution, and provide a clear structural representation."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining product 3 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the cyanide substitution site and nature in product 3 and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining product 3 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4a = "Sub-task 4a: Determine the detailed chemical structure of product 4 formed by treating product 3 with excess sodium hydride and 1,5-dibromopentane, explicitly describing the ring closure mechanism and resulting molecular framework, including atom-by-atom connectivity or a Lewis structure/SMILES notation."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3, answer3], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, determining detailed structure and ring closure mechanism of product 4, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    cot_instruction_4b = "Sub-task 4b: Label the carbons in the cyclization product (e.g., C1–C6), identify substituents (cyano, ethoxycarbonyl), and specify which carbons bear hydrogens, including explicit hydrogen counts per carbon."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "CoT"
    }
    thinking4b, answer4b = await cot_agent_4b([taskInfo, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4b.id}, labeling carbons and identifying substituents and hydrogens in product 4, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    cot_instruction_4c = "Sub-task 4c: Perform a stepwise Chain-of-Thought analysis of the ring closure mechanism and hydrogen placement on the cyclohexane ring, verifying chemical consistency (valence, substituent positions, hydrogen counts)."
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "CoT"
    }
    thinking4c, answer4c = await cot_agent_4c([taskInfo, thinking4b, answer4b], cot_instruction_4c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4c.id}, analyzing ring closure mechanism and hydrogen placement, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {
        "thinking": thinking4c,
        "answer": answer4c
    }
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])

    cot_reflect_instruction_4d = "Sub-task 4d: Conduct a Reflexion step to review and confirm the chemical structure and hydrogen assignments of product 4 for accuracy and consistency, correcting any errors found."
    cot_agent_4d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4d = self.max_round
    cot_inputs_4d = [taskInfo, thinking4c, answer4c]
    subtask_desc4d = {
        "subtask_id": "subtask_4d",
        "instruction": cot_reflect_instruction_4d,
        "context": ["user query", "thinking of subtask 4c", "answer of subtask 4c"],
        "agent_collaboration": "Reflexion"
    }
    thinking4d, answer4d = await cot_agent_4d(cot_inputs_4d, cot_reflect_instruction_4d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4d.id}, reviewing and confirming product 4 structure and hydrogen assignments, thinking: {thinking4d.content}; answer: {answer4d.content}")
    for i in range(N_max_4d):
        feedback, correct = await critic_agent_4d([taskInfo, thinking4d, answer4d], "please review the chemical structure and hydrogen assignments of product 4 for accuracy and consistency, and provide any corrections needed.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4d.extend([thinking4d, answer4d, feedback])
        thinking4d, answer4d = await cot_agent_4d(cot_inputs_4d, cot_reflect_instruction_4d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4d.id}, refining product 4 structure and hydrogen assignments, thinking: {thinking4d.content}; answer: {answer4d.content}")
    sub_tasks.append(f"Sub-task 4d output: thinking - {thinking4d.content}; answer - {answer4d.content}")
    subtask_desc4d['response'] = {
        "thinking": thinking4d,
        "answer": answer4d
    }
    logs.append(subtask_desc4d)
    print("Step 4d: ", sub_tasks[-1])

    cot_instruction_5a = "Sub-task 5a: Analyze the hydrogens on the cyclohexane ring of product 4, considering chair conformations, ring symmetry, and equivalence of axial and equatorial hydrogens, to identify chemically distinct hydrogen environments."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 4d", "answer of subtask 4d"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4d, answer4d], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, analyzing cyclohexane ring hydrogens, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])

    cot_instruction_5b = "Sub-task 5b: Analyze the hydrogens on the ethyl cyanoacetate substituent of product 4 separately, identifying chemically distinct hydrogen environments."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask 4d", "answer of subtask 4d"],
        "agent_collaboration": "CoT"
    }
    thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking4d, answer4d], cot_instruction_5b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5b.id}, analyzing ethyl cyanoacetate substituent hydrogens, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])

    cot_instruction_5c = "Sub-task 5c: Integrate the analyses from subtasks 5a and 5b to determine the total number of chemically distinct hydrogen signals expected in the 1H NMR spectrum of product 4, explicitly stating assumptions about symmetry and equivalence."
    cot_agent_5c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_instruction_5c,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "CoT"
    }
    thinking5c, answer5c = await cot_agent_5c([taskInfo, thinking5a, answer5a, thinking5b, answer5b], cot_instruction_5c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5c.id}, integrating hydrogen environment analyses, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {
        "thinking": thinking5c,
        "answer": answer5c
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])

    cot_sc_instruction_5d = "Sub-task 5d: Perform a Self-Consistency Chain-of-Thought or Reflexion step by generating multiple independent counts of chemically distinct hydrogens, comparing results, reconciling differences, and finalizing the count with clear justification."
    N5d = self.max_sc
    cot_agents_5d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5d)]
    possible_answers_5d = []
    thinkingmapping_5d = {}
    answermapping_5d = {}
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_sc_instruction_5d,
        "context": ["user query", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N5d):
        thinking5d, answer5d = await cot_agents_5d[i]([taskInfo, thinking5c, answer5c], cot_sc_instruction_5d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5d[i].id}, generating independent hydrogen counts, thinking: {thinking5d.content}; answer: {answer5d.content}")
        possible_answers_5d.append(answer5d.content)
        thinkingmapping_5d[answer5d.content] = thinking5d
        answermapping_5d[answer5d.content] = answer5d
    answer5d_content = Counter(possible_answers_5d).most_common(1)[0][0]
    thinking5d = thinkingmapping_5d[answer5d_content]
    answer5d = answermapping_5d[answer5d_content]
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {
        "thinking": thinking5d,
        "answer": answer5d
    }
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5d, answer5d, sub_tasks, agents)
    return final_answer, logs