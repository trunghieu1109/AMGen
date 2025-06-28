async def forward_81(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_instruction_1a = (
        "Sub-task 1a: Identify the specific reacting double bonds in cyclooctatetraene and maleic anhydride for the initial Diels–Alder reaction. "
        "Enumerate all plausible stereochemical (endo/exo) and regioisomeric product structures with explicit stereochemical descriptors (e.g., SMILES with stereochemistry). "
        "Provide mechanistic rationale for each isomer."
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
        thinking1a, answer1a = await cot_sc_agents_1a[i]([taskInfo], cot_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1a[i].id}, enumerating initial Diels–Alder products, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a
    answer1a_content = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[answer1a_content]
    answer1a = answermapping_1a[answer1a_content]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_instruction_1b = (
        "Sub-task 1b: Critically evaluate the stereochemical outcomes from subtask_1a using Diels–Alder selectivity rules (endo preference, steric hindrance, orbital interactions). "
        "Assign the most likely major isomer of product 1. Output the stereochemically explicit structure (SMILES/InChI) and a detailed stereochemical justification."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, evaluating stereochemical outcomes of product 1, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])

    cot_sc_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_instruction_2a = (
        "Sub-task 2a: Determine the chemical transformation of product 1 upon heating with methanol and sulfuric acid, focusing on the conversion of the anhydride to diester. "
        "Generate stereochemically explicit structures of product 2, preserving stereochemical information from product 1 and mapping stereochemistry onto the diester scaffold with machine-readable formats."
    )
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    for i in range(self.max_sc):
        thinking2a, answer2a = await cot_sc_agents_2a[i]([taskInfo, thinking1b, answer1b], cot_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2a[i].id}, determining product 2 structure, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])

    cot_instruction_2b = (
        "Sub-task 2b: Analyze and justify any stereochemical changes or retention during the methanolysis step, including conformational or configurational effects. "
        "Provide a clear stereochemical rationale and updated structural representation for product 2."
    )
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "CoT"
    }
    thinking2b, answer2b = await cot_agent_2b([taskInfo, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, analyzing stereochemical changes in methanolysis, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])

    debate_instruction_3a = (
        "Sub-task 3a: Enumerate all plausible regio- and stereoisomeric products formed by the Diels–Alder reaction of product 2 with cyclopentadiene, "
        "explicitly identifying reacting double bonds and possible endo/exo approaches. Generate stereochemically explicit structures (SMILES/InChI) for each candidate isomer."
    )
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3a = self.max_round
    all_thinking3a = [[] for _ in range(N_max_3a)]
    all_answer3a = [[] for _ in range(N_max_3a)]
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": debate_instruction_3a,
        "context": ["user query", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            if r == 0:
                thinking3a, answer3a = await agent([taskInfo, thinking2b, answer2b], debate_instruction_3a, r, is_sub_task=True)
            else:
                input_infos_3a = [taskInfo, thinking2b, answer2b] + all_thinking3a[r-1] + all_answer3a[r-1]
                thinking3a, answer3a = await agent(input_infos_3a, debate_instruction_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating Diels–Alder products with cyclopentadiene, thinking: {thinking3a.content}; answer: {answer3a.content}")
            all_thinking3a[r].append(thinking3a)
            all_answer3a[r].append(answer3a)
    thinking3a, answer3a = all_thinking3a[-1][0], all_answer3a[-1][0]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])

    debate_instruction_3b = (
        "Sub-task 3b: Conduct a detailed stereochemical analysis and debate of the candidate isomers from subtask_3a, applying Diels–Alder selectivity principles, steric and electronic effects, "
        "and conformational preferences to rank and select the major isomer of product 3. Provide a final stereochemically explicit structure with justification."
    )
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3b = self.max_round
    all_thinking3b = [[] for _ in range(N_max_3b)]
    all_answer3b = [[] for _ in range(N_max_3b)]
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": debate_instruction_3b,
        "context": ["user query", "thinking of subtask_3a", "answer of subtask_3a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking3b, answer3b = await agent([taskInfo, thinking3a, answer3a], debate_instruction_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking3a, answer3a] + all_thinking3b[r-1] + all_answer3b[r-1]
                thinking3b, answer3b = await agent(input_infos_3b, debate_instruction_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing and ranking candidate isomers, thinking: {thinking3b.content}; answer: {answer3b.content}")
            all_thinking3b[r].append(thinking3b)
            all_answer3b[r].append(answer3b)
    thinking3b, answer3b = all_thinking3b[-1][0], all_answer3b[-1][0]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])

    debate_instruction_4 = (
        "Sub-task 4: Compare the stereochemically explicit structure of the major isomer of product 3 from subtask_3b against the provided multiple-choice options by analyzing stereochemical descriptors and conformations. "
        "Identify the correct choice (A, B, C, or D) with a detailed rationale for the match."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask_3b", "answer of subtask_3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3b, answer3b], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3b, answer3b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing candidate structure with options, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the correct structure of the major isomer of product 3.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct structure, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
