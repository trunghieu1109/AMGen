async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Derive formal geometric representations of the regular dodecagon. "
        "Compute exact coordinates of all 12 vertices placed on the unit circle at multiples of 30 degrees. "
        "Enumerate all 12 sides and all 54 diagonals (all line segments connecting any two non-adjacent vertices), totaling 66 line segments. "
        "Represent these line segments as line equations or vector forms suitable for geometric computations. "
        "Validate correctness and consistency, ensuring no assumptions restrict the set of diagonals."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1, answer_1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, deriving geometric representations, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2a = (
        "Sub-task 2a: Enumerate all pairwise intersections of the 66 line segments (sides and diagonals) derived in Sub-task 1. "
        "For each pair of line segments, compute their intersection point if it exists and lies within both segments. "
        "Retain only intersection points that lie strictly inside the polygon or on its boundary, including vertices and interior intersection points formed by crossing diagonals. "
        "Output a comprehensive set of intersection points with their coordinates and the lines they lie on."
    )
    cot_sc_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_2a, answer_2a = await cot_sc_agents_2a[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2a[i].id}, enumerating all intersections, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
        possible_answers_2a.append(answer_2a.content)
        thinkingmapping_2a[answer_2a.content] = thinking_2a
        answermapping_2a[answer_2a.content] = answer_2a
    best_answer_2a = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking_2a = thinkingmapping_2a[best_answer_2a]
    answer_2a = answermapping_2a[best_answer_2a]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])

    cot_sc_instruction_2b = (
        "Sub-task 2b: Group all 66 line segments by their directions, which are multiples of 30 degrees due to the regular dodecagon's symmetry. "
        "For each of the 12 directions, identify all parallel lines (including sides and diagonals). "
        "For each pair of perpendicular directions (directions differing by 90 degrees modulo 180), consider all pairs of parallel lines in each direction. "
        "For each quadruple of lines (two parallel lines in one direction and two parallel lines in the perpendicular direction), compute their four intersection points. "
        "Verify whether these four points form a valid rectangle: closed and convex polygon, opposite sides parallel and equal in length, adjacent sides perpendicular, and all sides lie exactly on sides or diagonals of the dodecagon. "
        "Exclude degenerate or overlapping rectangles. Deduplicate rectangles by their corner sets to avoid double counting. "
        "Output a complete list of all valid rectangles formed by sides and diagonals, including those with vertices at interior intersection points."
    )
    cot_sc_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_2b, answer_2b = await cot_sc_agents_2b[i]([taskInfo, thinking_1, answer_1, thinking_2a, answer_2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2b[i].id}, enumerating and verifying rectangles, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
        possible_answers_2b.append(answer_2b.content)
        thinkingmapping_2b[answer_2b.content] = thinking_2b
        answermapping_2b[answer_2b.content] = answer_2b
    best_answer_2b = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[best_answer_2b]
    answer_2b = answermapping_2b[best_answer_2b]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])

    debate_instruction_3 = (
        "Sub-task 3: Classify the enumerated rectangles from Sub-task 2b into equivalence classes under the full symmetry group of the dodecagon (the dihedral group D12, including 12 rotations and 12 reflections). "
        "For each class, provide a formal combinatorial or geometric proof of the count of rectangles it contains, ensuring no double counting or omissions. "
        "Explicitly describe the action of the symmetry group on rectangles and verify orbit sizes. Avoid heuristic or ambiguous arguments. "
        "Output a rigorously justified count of rectangles per symmetry class, with clear documentation of the classification and counting method. "
        "Use debate among agents to refine and verify counts."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_rounds_3)]
    all_answer_3 = [[] for _ in range(N_rounds_3)]
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_2b, answer_2b], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_2b, answer_2b] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, classifying and counting rectangles, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 3: Finalize classification and counting of rectangles with formal proofs and symmetry considerations. Provide a consolidated count per symmetry class.", is_sub_task=True)
    agents.append(f"Final Decision agent, aggregating counts, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    cot_reflect_instruction_3_5 = (
        "Sub-task 3.5: Perform rigorous verification and reflexion on the counts obtained in Sub-task 3. "
        "Cross-verify counts by enumerating rectangles explicitly with their vertex sets and symmetry classes. "
        "Apply group theory arguments to confirm no double counting or omissions. Reconcile any discrepancies found during debate. "
        "Provide a final verified count of rectangles along with a detailed verification report. "
        "Ensure correctness and completeness before aggregation."
    )
    cot_reflect_agent_3_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_5 = self.max_round
    cot_inputs_3_5 = [taskInfo, thinking_3, answer_3]
    subtask_desc_3_5 = {
        "subtask_id": "subtask_3.5",
        "instruction": cot_reflect_instruction_3_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_5, answer_3_5 = await cot_reflect_agent_3_5(cot_inputs_3_5, cot_reflect_instruction_3_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent_3_5.id}, verifying counts, thinking: {thinking_3_5.content}; answer: {answer_3_5.content}")
    for i in range(N_max_3_5):
        feedback_3_5, correct_3_5 = await critic_agent_3_5([taskInfo, thinking_3_5, answer_3_5], "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_5.id}, feedback: {feedback_3_5.content}; correctness: {correct_3_5.content}")
        if correct_3_5.content == "True":
            break
        cot_inputs_3_5.extend([thinking_3_5, answer_3_5, feedback_3_5])
        thinking_3_5, answer_3_5 = await cot_reflect_agent_3_5(cot_inputs_3_5, cot_reflect_instruction_3_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent_3_5.id}, refining verification, thinking: {thinking_3_5.content}; answer: {answer_3_5.content}")
    sub_tasks.append(f"Sub-task 3.5 output: thinking - {thinking_3_5.content}; answer - {answer_3_5.content}")
    subtask_desc_3_5['response'] = {"thinking": thinking_3_5, "answer": answer_3_5}
    logs.append(subtask_desc_3_5)
    print("Step 3.5: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Aggregate the verified counts of rectangles from all symmetry classes obtained in Sub-task 3.5 to produce the total number of rectangles formed inside the dodecagon under the given constraints. "
        "Cross-check the final count against geometric intuition, alternative counting methods, or known results to confirm correctness. "
        "Provide a final answer along with a comprehensive verification summary documenting the entire reasoning, enumeration, symmetry analysis, and verification process."
    )
    cot_sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3.5", "answer of subtask 3.5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_4, answer_4 = await cot_sc_agents_4[i]([taskInfo, thinking_3_5, answer_3_5], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, aggregating and verifying final count, thinking: {thinking_4.content}; answer: {answer_4.content}")
        possible_answers_4.append(answer_4.content)
        thinkingmapping_4[answer_4.content] = thinking_4
        answermapping_4[answer_4.content] = answer_4
    best_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking_4 = thinkingmapping_4[best_answer_4]
    answer_4 = answermapping_4[best_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs
