async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify and formalize the fundamental parameters and variables from the problem statement. "
        "Define the representation of a two-digit number n in base b as n = x*b + y, with digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "Define the sum of digits s = x + y and express the b-eautiful condition as s = √n. "
        "Translate this into the key equation (x + y)^2 = x*b + y. Carefully consider digit bounds and base constraints, explicitly stating assumptions and avoiding contradictions. "
        "This subtask lays the mathematical foundation for all subsequent analysis."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before Sub-task 1: {subtask_desc1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Derive and rigorously validate the formal problem condition from Sub-task 1. "
        "Confirm that the equation (x + y)^2 = x*b + y correctly characterizes b-eautiful numbers under the digit constraints and base restrictions. "
        "Analyze the implications of these constraints on possible values of x, y, and b, including the range of n (b ≤ n ≤ b^2 - 1). "
        "Explicitly verify that the problem conditions are consistent and complete, ensuring no overlooked edge cases. "
        "This step ensures the correctness and completeness of the mathematical model before enumeration."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before Sub-task 2: {subtask_desc2}")
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, validating problem representation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3a = (
        "Sub-task 3a: Perform algebraic analysis to derive constraints on the sum of digits s = x + y and its relation to base b, "
        "aiming to reduce the search space for enumeration. Specifically, analyze the equation (x + y)^2 = x*b + y to express y in terms of x, b, and s, "
        "and deduce bounds on s and x that must hold for integer digit solutions. Use these constraints to prune impossible digit pairs and avoid brute-force over the entire digit range. "
        "Document all algebraic deductions and ensure they are logically sound and complete to support efficient enumeration."
    )
    N_sc_3a = self.max_sc
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3a)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before Sub-task 3a: {subtask_desc3a}")
    for i in range(N_sc_3a):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, algebraic analysis for enumeration pruning, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    best_answer_3a = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[best_answer_3a]
    answer3a = answermapping_3a[best_answer_3a]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_instruction_3b = (
        "Sub-task 3b: Enumerate all valid digit pairs (x,y) for each base b in a reasonable range (2 to 50), "
        "strictly within the algebraically derived bounds from Sub-task 3a. For each candidate (x,y), verify the equation (x + y)^2 = x*b + y and digit constraints. "
        "Generate a comprehensive mapping from each base b to the full list of b-eautiful numbers and their counts. Implement checks to avoid double counting and ensure completeness. "
        "Output detailed intermediate data including solution lists per base to facilitate verification."
    )
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before Sub-task 3b: {subtask_desc3b}")
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, enumerating valid digit pairs per base, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    debate_instruction_4a = (
        "Sub-task 4a: Verify the enumeration results from Sub-task 3b by cross-checking counts against algebraic bounds and constraints. "
        "Confirm the uniqueness of each solution to prevent duplicates. Analyze the distribution of solutions per base and identify any anomalies or inconsistencies. "
        "If discrepancies are found, request re-examination or refinement of enumeration. This verification step ensures the reliability of the data before final analysis."
    )
    debate_agents_4a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_4a = self.max_round
    all_thinking_4a = [[] for _ in range(N_rounds_4a)]
    all_answer_4a = [[] for _ in range(N_rounds_4a)]
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": debate_instruction_4a,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before Sub-task 4a: {subtask_desc4a}")
    for r in range(N_rounds_4a):
        for i, agent in enumerate(debate_agents_4a):
            if r == 0:
                thinking4a, answer4a = await agent([taskInfo, thinking3b, answer3b], debate_instruction_4a, r, is_sub_task=True)
            else:
                input_infos_4a = [taskInfo, thinking3b, answer3b] + all_thinking_4a[r-1] + all_answer_4a[r-1]
                thinking4a, answer4a = await agent(input_infos_4a, debate_instruction_4a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying enumeration results, thinking: {thinking4a.content}; answer: {answer4a.content}")
            all_thinking_4a[r].append(thinking4a)
            all_answer_4a[r].append(answer4a)
    sub_tasks.append(f"Sub-task 4a output: thinking - {all_thinking_4a[-1][0].content}; answer - {all_answer_4a[-1][0].content}")
    subtask_desc4a['response'] = {
        "thinking": all_thinking_4a[-1][0],
        "answer": all_answer_4a[-1][0]
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    reflexion_instruction_4b = (
        "Sub-task 4b: Analyze the verified enumeration data to identify the smallest base b ≥ 2 for which the number of b-eautiful integers exceeds ten. "
        "Provide a rigorous justification or minimality proof, combining algebraic reasoning and enumeration evidence. Present the final answer alongside the verification results, "
        "ensuring no valid solutions are missed and no invalid solutions are included. This subtask synthesizes all prior work into a conclusive, well-supported solution to the original query."
    )
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4b = self.max_round
    cot_inputs_4b = [taskInfo] + all_thinking_4a[-1] + all_answer_4a[-1]
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": reflexion_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before Sub-task 4b: {subtask_desc4b}")
    thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, reflexion_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, analyzing minimal base and justification, thinking: {thinking4b.content}; answer: {answer4b.content}")
    for i in range(N_max_4b):
        feedback4b, correct4b = await critic_agent_4b([taskInfo, thinking4b, answer4b],
                                                    "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.",
                                                    i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback, thinking: {feedback4b.content}; answer: {correct4b.content}")
        if correct4b.content.strip() == "True":
            break
        cot_inputs_4b.extend([thinking4b, answer4b, feedback4b])
        thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, reflexion_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining minimal base analysis, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4b, answer4b, sub_tasks, agents)
    return final_answer, logs
