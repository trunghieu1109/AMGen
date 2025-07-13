async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Define variables a11, a12, a13 for the first row and a21, a22, a23 for the second row. "
        "Express the two 3-digit row numbers as N1 = 100*a11 + 10*a12 + a13 and N2 = 100*a21 + 10*a22 + a23, with the equation N1 + N2 = 999. "
        "Define the three 2-digit column numbers as C1 = 10*a11 + a21, C2 = 10*a12 + a22, C3 = 10*a13 + a23, with the equation C1 + C2 + C3 = 99. "
        "Clarify assumptions: digits range 0-9, leading zeros allowed, digits may repeat. "
        "Avoid unsupported constraints. Provide a precise mathematical framework for subsequent subtasks."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formalizing variables and equations, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    cot_sc_instruction_1 = (
        "Sub-task 2: Enumerate all digit assignments (a11, a12, a13, a21, a22, a23) satisfying the row sum N1 + N2 = 999. "
        "Analyze addition column-wise with carries c1, c2, c3, explicitly track carry values for each digit addition. "
        "Use systematic enumeration or algebraic characterization to ensure completeness and correctness. "
        "Do not filter by column sums yet. Output a comprehensive set of candidate digit assignments with carry values."
    )
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, enumerate row sum candidates with carry, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1.append(answer_i.content)
        thinkingmapping_1[answer_i.content] = thinking_i
        answermapping_1[answer_i.content] = answer_i
    best_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking_1 = thinkingmapping_1[best_answer_1]
    answer_1 = answermapping_1[best_answer_1]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)

    cot_sc_instruction_2a = (
        "Sub-task 3a: From candidate digit assignments and carry values in Sub-task 2, enumerate all possible pairs (S1, S2) where S1 = a11 + a12 + a13 and S2 = a21 + a22 + a23, "
        "such that 10*S1 + S2 = 99 holds. Enforce consistency with carry values from row sum addition to filter invalid pairs. "
        "Output the refined set of valid (S1, S2) pairs with their carry patterns."
    )
    N_sc_2a = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2a)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc_2a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2a):
        thinking_2a, answer_2a = await cot_agents_2a[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, filter valid (S1,S2) pairs with carry consistency, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
        possible_answers_2a.append(answer_2a.content)
        thinkingmapping_2a[answer_2a.content] = thinking_2a
        answermapping_2a[answer_2a.content] = answer_2a
    best_answer_2a = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking_2a = thinkingmapping_2a[best_answer_2a]
    answer_2a = answermapping_2a[best_answer_2a]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)

    cot_instruction_2b = (
        "Sub-task 3b: For each valid (S1, S2) pair and carry pattern from Sub-task 3a, generate all digit triples (a11, a12, a13) summing to S1 with correct carry propagation. "
        "Compute complementary bottom row digits (a21, a22, a23) accordingly. Verify that the three column numbers C1, C2, C3 formed by vertical pairs satisfy C1 + C2 + C3 = 99 exactly, including carry consistency in vertical addition. "
        "Output the final refined set of valid digit assignments satisfying both sum constraints with full carry validation."
    )
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc_2b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_2b,
        "context": ["user query", thinking_2a.content, answer_2a.content],
        "agent_collaboration": "CoT"
    }
    thinking_2b, answer_2b = await cot_agent_2b([taskInfo, thinking_2a, answer_2a], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, generate and verify full digit assignments with vertical sum, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)

    debate_instruction_3 = (
        "Sub-task 4: Validation Agent performs systematic brute-force or constraint-based verification over all candidate digit assignments from Sub-task 3b. "
        "Verify each candidate satisfies row sum 999 with correct carry, column sum 99 with correct carry, digits in 0-9, leading zeros and repetition allowed. "
        "Count total valid solutions and cross-validate with known examples. Output final verified count with representative examples and explicit carry checks."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_rounds_3)]
    all_answer_3 = [[] for _ in range(N_rounds_3)]
    subtask_desc_3 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking_2b.content, answer_2b.content],
        "agent_collaboration": "Debate | Reflexion"
    }
    for r in range(N_rounds_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_2b, answer_2b], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_2b, answer_2b] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validate candidates, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)

    cot_agent_reflect_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    final_reflect_instruction_3 = (
        "Sub-task 4 continued: Reflect on debate outputs and finalize the validated set of digit assignments. "
        "Confirm correctness and completeness of the count and provide representative examples with carry checks."
    )
    thinking_3_reflect, answer_3_reflect = await cot_agent_reflect_3([taskInfo, thinking_2b, answer_2b] + all_thinking_3[-1] + all_answer_3[-1], final_reflect_instruction_3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect_3.id}, finalize validation, thinking: {thinking_3_reflect.content}; answer: {answer_3_reflect.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3_reflect.content}; answer - {answer_3_reflect.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3_reflect, "answer": answer_3_reflect}
    logs.append(subtask_desc_3)

    cot_instruction_4 = (
        "Sub-task 5: Aggregate validated digit assignments and counts from Sub-task 4 to produce the final answer: total number of valid digit assignments. "
        "Provide detailed justification including sample valid assignments, carry values, and verification results. "
        "Perform sanity checks by sampling representative solutions and confirming sums and carry consistency. "
        "Avoid duplicates or omissions. Present final answer with comprehensive verification."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking_3_reflect.content, answer_3_reflect.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    thinking_4, answer_4 = await cot_agent_4([taskInfo, thinking_3_reflect, answer_3_reflect], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, aggregate final count and verify, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)

    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    for i, st in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", st)
    return final_answer, logs
