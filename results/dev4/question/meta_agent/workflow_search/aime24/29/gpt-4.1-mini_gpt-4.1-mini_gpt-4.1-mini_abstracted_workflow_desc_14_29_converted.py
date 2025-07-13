async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Extract and formalize all given information and constraints from the problem statement. "
        "Clarify the meaning of color uniformity per row and column, explicitly state assumptions about empty rows and columns, "
        "and define the maximality condition precisely. Emphasize chip indistinguishability and focus on pattern-level constraints. "
        "Avoid unverified assumptions; clearly state any assumptions as conditions to be tested later."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extract and formalize constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze compatibility conditions between row and column color assignments induced by intersection cells. "
        "Formalize constraints as a system on row and column color vectors, determine how cell colors depend on these assignments. "
        "Clarify maximality condition ensuring no empty cell can be filled without violating uniformity or uniqueness. "
        "Focus on pattern-level constraints and explicitly identify how maximality restricts placements."
    )
    N2 = self.max_sc
    cot_sc_agents_2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N2)
    ]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, analyze compatibility conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    best_answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer2]
    answer2 = answermapping_2[best_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Derive a formal combinatorial representation of the problem, such as a bipartite graph or matrix with row and column color assignments. "
        "Express maximality as a maximal matching or coloring problem. Identify parameters like number of rows and columns assigned black or white and how these determine placement patterns. "
        "Develop a general combinatorial model capturing all valid maximal placements without premature simplifications. "
        "Ensure model incorporates constraints from previous subtasks."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2.content, answer2.content], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, derive combinatorial model, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Compute the number of valid maximal placements using the derived combinatorial model. "
        "Enumerate all possible row and column color assignments consistent with constraints and maximality, count resulting placement patterns, and simplify formula if possible. "
        "Verify count respects chip availability and grid size. Avoid double counting or ignoring indistinguishability. Provide detailed numeric answer and enumeration method. "
        "This output must be passed fully and explicitly to subsequent verification subtasks."
    )
    N4 = self.max_sc
    cot_sc_agents_4 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N4)
    ]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4):
        thinking4, answer4 = await cot_sc_agents_4[i]([taskInfo, thinking3.content, answer3.content], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, compute valid maximal placements, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    best_answer4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[best_answer4]
    answer4 = answermapping_4[best_answer4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_5a = (
        "Sub-task 5a: Verify correctness of the computed count from Subtask 4 by cross-checking reasoning, assumptions, and final formula. "
        "Perform alternative reasoning and validate maximality and chip-count constraints explicitly. "
        "Explicitly reference numeric count and enumeration method from Subtask 4; do not ignore or overwrite it."
    )
    debate_agents_5a = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_5a = self.max_round
    all_thinking5a = [[] for _ in range(N_max_5a)]
    all_answer5a = [[] for _ in range(N_max_5a)]
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": debate_instr_5a,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5a):
        for i, agent in enumerate(debate_agents_5a):
            if r == 0:
                thinking5a, answer5a = await agent([taskInfo, thinking4.content, answer4.content], debate_instr_5a, r, is_sub_task=True)
            else:
                input_infos_5a = [taskInfo, thinking4.content, answer4.content] + all_thinking5a[r-1] + all_answer5a[r-1]
                thinking5a, answer5a = await agent(input_infos_5a, debate_instr_5a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verify count correctness, thinking: {thinking5a.content}; answer: {answer5a.content}")
            all_thinking5a[r].append(thinking5a)
            all_answer5a[r].append(answer5a)
    sub_tasks.append(f"Sub-task 5a output: thinking - {all_thinking5a[-1][0].content}; answer - {all_answer5a[-1][0].content}")
    subtask_desc5a['response'] = {
        "thinking": all_thinking5a[-1][0],
        "answer": all_answer5a[-1][0]
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])

    cot_sc_instruction_5b = (
        "Sub-task 5b: Empirically validate the counting model by enumerating all valid maximal placements on smaller grids (1x1, 2x2, 3x3) using brute-force or automated checks. "
        "Compare these results with predictions of the combinatorial model and counting formula. "
        "Use discrepancies to detect overcounting or undercounting and refine assumptions if needed. "
        "Provide concrete evidence supporting or challenging the refined count."
    )
    N5b = self.max_sc
    cot_sc_agents_5b = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N5b)
    ]
    possible_answers_5b = []
    thinkingmapping_5b = {}
    answermapping_5b = {}
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_sc_instruction_5b,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N5b):
        thinking5b, answer5b = await cot_sc_agents_5b[i]([taskInfo, thinking4.content, answer4.content], cot_sc_instruction_5b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_5b[i].id}, empirical validation on small grids, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers_5b.append(answer5b.content)
        thinkingmapping_5b[answer5b.content] = thinking5b
        answermapping_5b[answer5b.content] = answer5b
    best_answer5b = Counter(possible_answers_5b).most_common(1)[0][0]
    thinking5b = thinkingmapping_5b[best_answer5b]
    answer5b = answermapping_5b[best_answer5b]
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])

    debate_instr_5c = (
        "Sub-task 5c: Conduct a structured debate with assigned roles: one agent defends the refined enumeration and numeric count from Subtask 4, "
        "another agent challenges it critically focusing on maximality, chip-count constraints, and empirical validation results. "
        "The final decision agent synthesizes the discussion, explicitly comparing numeric results and symbolic formulas, justifying why the refined count is correct and why naive counts are overcounts. "
        "Mandate critical numeric evaluation, not just verbal reasoning."
    )
    debate_agents_5c = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_5c = self.max_round
    all_thinking5c = [[] for _ in range(N_max_5c)]
    all_answer5c = [[] for _ in range(N_max_5c)]
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": debate_instr_5c,
        "context": ["user query", thinking4.content, answer4.content, all_thinking5a[-1][0].content, all_answer5a[-1][0].content, thinking5b.content, answer5b.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5c):
        for i, agent in enumerate(debate_agents_5c):
            if r == 0:
                thinking5c, answer5c = await agent(
                    [taskInfo, thinking4.content, answer4.content, all_thinking5a[-1][0].content, all_answer5a[-1][0].content, thinking5b.content, answer5b.content],
                    debate_instr_5c, r, is_sub_task=True)
            else:
                input_infos_5c = [taskInfo, thinking4.content, answer4.content, all_thinking5a[-1][0].content, all_answer5a[-1][0].content, thinking5b.content, answer5b.content] + all_thinking5c[r-1] + all_answer5c[r-1]
                thinking5c, answer5c = await agent(input_infos_5c, debate_instr_5c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, structured debate on refined count, thinking: {thinking5c.content}; answer: {answer5c.content}")
            all_thinking5c[r].append(thinking5c)
            all_answer5c[r].append(answer5c)
    sub_tasks.append(f"Sub-task 5c output: thinking - {all_thinking5c[-1][0].content}; answer - {all_answer5c[-1][0].content}")
    subtask_desc5c['response'] = {
        "thinking": all_thinking5c[-1][0],
        "answer": all_answer5c[-1][0]
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])

    reflect_instruction_6 = (
        "Sub-task 6: Synthesize verification feedback and debate conclusions to finalize the answer. "
        "Confirm no contradictions remain, reconcile conflicting results, and produce a well-justified final numeric answer with explicit reasoning. "
        "Document all assumptions, validation steps, and justifications clearly. "
        "Ensure final answer is consistent with all prior subtasks and explicitly references refined enumeration and verification outcomes."
    )
    cot_reflect_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": reflect_instruction_6,
        "context": ["user query", thinking4.content, answer4.content, all_thinking5a[-1][0].content, all_answer5a[-1][0].content, thinking5b.content, answer5b.content, all_thinking5c[-1][0].content, all_answer5c[-1][0].content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_reflect_agent_6(
        [taskInfo, thinking4.content, answer4.content, all_thinking5a[-1][0].content, all_answer5a[-1][0].content, thinking5b.content, answer5b.content, all_thinking5c[-1][0].content, all_answer5c[-1][0].content],
        reflect_instruction_6, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent_6.id}, synthesize final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
