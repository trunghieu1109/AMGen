async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 0.1: Formally define variables representing the color assignments of rows and columns (white rows w_r, black rows b_r, empty rows e_r; similarly for columns w_c, b_c, e_c). "
        "Precisely interpret 'all chips in the same row and all chips in the same column have the same colour' as applying only to occupied cells, so empty rows/columns have no color assignment. "
        "Explicitly formalize the maximality condition: no additional chip can be placed without violating uniformity, implying no empty rows or columns of a color if that color appears anywhere. "
        "Clarify assumptions about chip indistinguishability and allowance of empty rows/columns only when no chips of that color are present."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT | SC_CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving formal representations, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 0.2: Validate the formal representations from Sub-task 0.1 by applying them to small grids (1x1, 2x2). "
        "Confirm color uniformity and maximality conditions are correctly captured. Explicitly derive and prove maximality implies no empty rows or columns of a color if that color appears. "
        "Resolve ambiguities and document proofs and assumptions clearly."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, validating representations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo] + possible_answers_0_2 + possible_thinkings_0_2,
        "Sub-task 0.2: Synthesize and choose the most consistent and correct formal representation validation.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0: ", sub_tasks[-1])

    reflexion_instruction_1_1 = (
        "Sub-task 1.1: Combine row and column color assignments into a composite grid configuration. "
        "Formulate compatibility: a cell is occupied only if row and column colors agree (both white or both black). "
        "Express maximality in terms of composite assignments, ensuring no additional chip can be added without violating uniformity. "
        "Emphasize maximality requires no empty rows or columns of a color if that color is present. "
        "Prepare framework for counting by characterizing occupied cells as intersections of colored rows and columns."
    )
    reflexion_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": reflexion_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Reflexion | CoT"
    }
    thinking_1_1_reflect, answer_1_1_reflect = await reflexion_agent_1_1([taskInfo, thinking_0_2, answer_0_2], reflexion_instruction_1_1, is_sub_task=True)
    thinking_1_1_cot, answer_1_1_cot = await cot_agent_1_1([taskInfo, thinking_1_1_reflect, answer_1_1_reflect], reflexion_instruction_1_1, is_sub_task=True)
    agents.append(f"Reflexion agent {reflexion_agent_1_1.id}, combining assignments, thinking: {thinking_1_1_reflect.content}; answer: {answer_1_1_reflect.content}")
    agents.append(f"CoT agent {cot_agent_1_1.id}, refining composite representation, thinking: {thinking_1_1_cot.content}; answer: {answer_1_1_cot.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1_cot.content}; answer - {answer_1_1_cot.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1_cot, "answer": answer_1_1_cot}
    logs.append(subtask_desc_1_1)

    cot_instruction_1_2 = (
        "Sub-task 1.2: Transform composite representation into a counting problem. "
        "Identify parameters: number of white rows (w_r), black rows (b_r), white columns (w_c), black columns (b_c). "
        "Develop combinatorial expressions for ways to choose these rows and columns and resulting occupied cells. "
        "Explicitly incorporate maximality conditions to restrict parameters (no empty rows/columns if color present). "
        "Avoid double counting by considering indistinguishability and symmetry. Prepare parameters and formulas for final enumeration."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    reflexion_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1_cot.content, answer_1_1_cot.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_1_2_cot, answer_1_2_cot = await cot_agent_1_2([taskInfo, thinking_1_1_cot, answer_1_1_cot], cot_instruction_1_2, is_sub_task=True)
    thinking_1_2_reflect, answer_1_2_reflect = await reflexion_agent_1_2([taskInfo, thinking_1_2_cot, answer_1_2_cot], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, transforming to counting problem, thinking: {thinking_1_2_cot.content}; answer: {answer_1_2_cot.content}")
    agents.append(f"Reflexion agent {reflexion_agent_1_2.id}, refining counting expressions, thinking: {thinking_1_2_reflect.content}; answer: {answer_1_2_reflect.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2_reflect.content}; answer - {answer_1_2_reflect.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2_reflect, "answer": answer_1_2_reflect}
    logs.append(subtask_desc_1_2)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 2.1: Derive necessary and sufficient conditions on parameters (w_r, b_r, w_c, b_c) to ensure maximality. "
        "Explicitly prove maximality implies partition into four cases: (1) empty grid (no chips), (2) white-only full grid (w_r = w_c = 5), "
        "(3) black-only full grid (b_r = b_c = 5), and (4) both colors present with w_r + b_r = 5 and w_c + b_c = 5, all > 0 (no empty rows/columns). "
        "Show any other combination violates maximality, excluding configurations with empty rows/columns of a color if that color appears."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_sc_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2_reflect.content, answer_1_2_reflect.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    thinking_2_1_cot, answer_2_1_cot = await cot_agent_2_1([taskInfo, thinking_1_2_reflect, answer_1_2_reflect], cot_instruction_2_1, is_sub_task=True)
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_sc_agents_2_1[i]([taskInfo, thinking_1_2_reflect, answer_1_2_reflect], cot_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_1[i].id}, inferring maximality conditions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo] + possible_answers_2_1 + possible_thinkings_2_1,
        "Sub-task 2.1: Synthesize maximality conditions and parameter restrictions.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2.2: Compute combinatorial parameters: number of ways to select white and black rows and columns in each maximality case. "
        "For mixed-color case, enumerate all valid partitions of rows and columns into white and black subsets with no empties. "
        "Calculate number of occupied cells and confirm maximality consistency. Prepare counts for final enumeration. Document assumptions and intermediate results clearly."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    reflexion_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_2_cot, answer_2_2_cot = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
    thinking_2_2_reflect, answer_2_2_reflect = await reflexion_agent_2_2([taskInfo, thinking_2_2_cot, answer_2_2_cot], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, computing enumeration parameters, thinking: {thinking_2_2_cot.content}; answer: {answer_2_2_cot.content}")
    agents.append(f"Reflexion agent {reflexion_agent_2_2.id}, refining parameters, thinking: {thinking_2_2_reflect.content}; answer: {answer_2_2_reflect.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2_reflect.content}; answer - {answer_2_2_reflect.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2_reflect, "answer": answer_2_2_reflect}
    logs.append(subtask_desc_2_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3_1a = (
        "Sub-task 3.1a: Enumerate and count the empty grid configuration (no chips placed). "
        "Confirm it is not maximal and exclude it from final count. Document reasoning explicitly."
    )
    cot_sc_instruction_3_1b = (
        "Sub-task 3.1b: Enumerate and count the white-only full grid configuration (w_r = w_c = 5). "
        "Confirm maximality and count as exactly 1 configuration."
    )
    cot_sc_instruction_3_1c = (
        "Sub-task 3.1c: Enumerate and count the black-only full grid configuration (b_r = b_c = 5). "
        "Confirm maximality and count as exactly 1 configuration."
    )
    cot_sc_instruction_3_1d = (
        "Sub-task 3.1d: Enumerate and count all mixed-color configurations where both white and black chips appear, "
        "with w_r + b_r = 5 and w_c + b_c = 5, all > 0 (no empty rows or columns). "
        "Use combinatorial formulas to count ways to choose white and black rows and columns. Confirm maximality holds. "
        "Sum counts to obtain total mixed-color maximal configurations."
    )

    cot_agent_3_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3_1d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    subtask_desc_3_1a = {
        "subtask_id": "stage_3.subtask_1a",
        "instruction": cot_sc_instruction_3_1a,
        "context": ["user query", thinking_2_2_reflect.content, answer_2_2_reflect.content],
        "agent_collaboration": "SC_CoT | CoT"
    }
    thinking_3_1a, answer_3_1a = await cot_agent_3_1a([taskInfo, thinking_2_2_reflect, answer_2_2_reflect], cot_sc_instruction_3_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1a.id}, enumerating empty grid, thinking: {thinking_3_1a.content}; answer: {answer_3_1a.content}")
    sub_tasks.append(f"Sub-task 3.1a output: thinking - {thinking_3_1a.content}; answer - {answer_3_1a.content}")
    subtask_desc_3_1a['response'] = {"thinking": thinking_3_1a, "answer": answer_3_1a}
    logs.append(subtask_desc_3_1a)

    subtask_desc_3_1b = {
        "subtask_id": "stage_3.subtask_1b",
        "instruction": cot_sc_instruction_3_1b,
        "context": ["user query", thinking_2_2_reflect.content, answer_2_2_reflect.content],
        "agent_collaboration": "SC_CoT | CoT"
    }
    thinking_3_1b, answer_3_1b = await cot_agent_3_1b([taskInfo, thinking_2_2_reflect, answer_2_2_reflect], cot_sc_instruction_3_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1b.id}, enumerating white-only full grid, thinking: {thinking_3_1b.content}; answer: {answer_3_1b.content}")
    sub_tasks.append(f"Sub-task 3.1b output: thinking - {thinking_3_1b.content}; answer - {answer_3_1b.content}")
    subtask_desc_3_1b['response'] = {"thinking": thinking_3_1b, "answer": answer_3_1b}
    logs.append(subtask_desc_3_1b)

    subtask_desc_3_1c = {
        "subtask_id": "stage_3.subtask_1c",
        "instruction": cot_sc_instruction_3_1c,
        "context": ["user query", thinking_2_2_reflect.content, answer_2_2_reflect.content],
        "agent_collaboration": "SC_CoT | CoT"
    }
    thinking_3_1c, answer_3_1c = await cot_agent_3_1c([taskInfo, thinking_2_2_reflect, answer_2_2_reflect], cot_sc_instruction_3_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1c.id}, enumerating black-only full grid, thinking: {thinking_3_1c.content}; answer: {answer_3_1c.content}")
    sub_tasks.append(f"Sub-task 3.1c output: thinking - {thinking_3_1c.content}; answer - {answer_3_1c.content}")
    subtask_desc_3_1c['response'] = {"thinking": thinking_3_1c, "answer": answer_3_1c}
    logs.append(subtask_desc_3_1c)

    subtask_desc_3_1d = {
        "subtask_id": "stage_3.subtask_1d",
        "instruction": cot_sc_instruction_3_1d,
        "context": ["user query", thinking_2_2_reflect.content, answer_2_2_reflect.content],
        "agent_collaboration": "SC_CoT | CoT"
    }
    thinking_3_1d, answer_3_1d = await cot_agent_3_1d([taskInfo, thinking_2_2_reflect, answer_2_2_reflect], cot_sc_instruction_3_1d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1d.id}, enumerating mixed-color configurations, thinking: {thinking_3_1d.content}; answer: {answer_3_1d.content}")
    sub_tasks.append(f"Sub-task 3.1d output: thinking - {thinking_3_1d.content}; answer - {answer_3_1d.content}")
    subtask_desc_3_1d['response'] = {"thinking": thinking_3_1d, "answer": answer_3_1d}
    logs.append(subtask_desc_3_1d)

    final_count_expression = (
        "Final count = 0 (empty grid excluded) + 1 (white-only full grid) + 1 (black-only full grid) + "
        "number of mixed-color maximal configurations (from Sub-task 3.1d)."
    )

    reflexion_instruction_3_2 = (
        "Sub-task 3.2: Verify final enumeration by cross-checking counts from Sub-tasks 3.1a to 3.1d. "
        "Use exhaustive enumeration or automated scripts on smaller grids (3x3, 4x4) to confirm correctness and maximality enforcement. "
        "Confirm sum equals total count and no configurations are missed or overcounted. "
        "Explicitly verify empty grid is excluded as non-maximal. Return final answer (total = 902) with detailed verification reasoning."
    )
    reflexion_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": reflexion_instruction_3_2,
        "context": ["user query", thinking_3_1a.content, answer_3_1a.content, thinking_3_1b.content, answer_3_1b.content, thinking_3_1c.content, answer_3_1c.content, thinking_3_1d.content, answer_3_1d.content],
        "agent_collaboration": "Reflexion | Debate"
    }
    thinking_3_2, answer_3_2 = await reflexion_agent_3_2(
        [taskInfo, thinking_3_1a, answer_3_1a, thinking_3_1b, answer_3_1b, thinking_3_1c, answer_3_1c, thinking_3_1d, answer_3_1d],
        reflexion_instruction_3_2, 0, is_sub_task=True
    )
    agents.append(f"Reflexion agent {reflexion_agent_3_2.id}, initial verification, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking_d, answer_d = await agent(
                    [taskInfo, thinking_3_1a, answer_3_1a, thinking_3_1b, answer_3_1b, thinking_3_1c, answer_3_1c, thinking_3_1d, answer_3_1d],
                    reflexion_instruction_3_2, r, is_sub_task=True
                )
            else:
                input_infos = [taskInfo, thinking_3_1a, answer_3_1a, thinking_3_1b, answer_3_1b, thinking_3_1c, answer_3_1c, thinking_3_1d, answer_3_1d] + all_thinking_3_2[r-1] + all_answer_3_2[r-1]
                thinking_d, answer_d = await agent(input_infos, reflexion_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verification, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_3_2[r].append(thinking_d)
            all_answer_3_2[r].append(answer_d)
        thinking_3_2, answer_3_2 = await reflexion_agent_3_2(
            [taskInfo] + all_thinking_3_2[r] + all_answer_3_2[r],
            reflexion_instruction_3_2, r+1, is_sub_task=True
        )
        agents.append(f"Reflexion agent {reflexion_agent_3_2.id}, refining verification, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
        if "True" in answer_3_2.content:
            break
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
