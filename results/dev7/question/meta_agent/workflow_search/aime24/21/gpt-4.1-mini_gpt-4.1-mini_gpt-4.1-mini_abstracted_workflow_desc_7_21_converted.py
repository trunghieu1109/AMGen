async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Compute the complete geometric model of the regular dodecagon including all 12 vertices, all sides, and all diagonals connecting any two distinct vertices. "
        "Calculate all intersection points formed by these sides and diagonals inside the polygon (not just polygon vertices). "
        "Construct a planar graph where nodes represent all these points (vertices and intersection points), and edges represent maximal straight line segments between consecutive intersection points along sides or diagonals. "
        "Ensure numerical precision and completeness, avoiding assumptions that rectangle vertices lie only on polygon vertices. "
        "This enriched geometric data structure will serve as the foundation for all subsequent analysis."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, computing full planar graph with intersection points, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the planar graph edges obtained from Subtask 1 by labeling each edge with its direction (angle modulo 180 degrees). "
        "Identify all pairs of edges that are parallel and all pairs that are perpendicular, considering subdivided edges between intersection points, not just chords between polygon vertices. "
        "Compute precise angles and apply tolerances for floating-point errors. Record adjacency relations between edges and nodes to facilitate cycle enumeration. "
        "Avoid limiting analysis to original polygon edges or vertex-to-vertex diagonals only."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing directions, parallelism, and perpendicularity, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo] + possible_answers_2 + possible_thinkings_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct analysis of edge directions, parallelism, and perpendicularity.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)

    cot_sc_instruction_3 = (
        "Sub-task 3: Enumerate all simple 4-cycles (closed loops of length 4) in the planar graph constructed in Subtask 1. "
        "For each 4-cycle, verify rectangle properties: (a) opposite edges are parallel and equal in length, (b) adjacent edges are perpendicular, and (c) the polygon formed is convex with nonzero area. "
        "Use direction and adjacency information from Subtask 2 to prune candidates efficiently. Include rectangles formed by interior intersection points, not just polygon vertices. "
        "Avoid degenerate or self-intersecting quadrilaterals. Ensure enumeration is exhaustive and precise to avoid undercounting."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, enumerating and verifying rectangles, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo] + possible_answers_3 + possible_thinkings_3,
        "Sub-task 3: Synthesize and choose the most consistent and correct enumeration of rectangles.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)

    cot_sc_instruction_4 = (
        "Sub-task 4: Remove duplicate rectangles from the enumeration in Subtask 3 by representing each rectangle in a canonical form (e.g., sorted vertex coordinates or edge labels). "
        "Use the symmetry group of the regular dodecagon to cross-validate uniqueness and completeness. Confirm all rectangles satisfy problem constraints strictly, including that each side lies on a side or diagonal of the polygon. "
        "Identify and exclude any degenerate or overlapping rectangles."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT | Debate"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, verifying uniqueness and removing duplicates, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)

    debate_instr_4 = (
        "Sub-task 4 Debate: Given the enumerated rectangles and uniqueness verification, debate among agents to challenge and confirm the completeness and correctness of the rectangle set. "
        "Consider symmetry, possible missed cases, and duplicates. Provide a consensus refined set of unique rectangles."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4_d, answer4_d = await agent([taskInfo, thinking3, answer3] + possible_thinkings_4 + possible_answers_4, debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_d, answer4_d = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating uniqueness and completeness, thinking: {thinking4_d.content}; answer: {answer4_d.content}")
            all_thinking4[r].append(thinking4_d)
            all_answer4[r].append(answer4_d)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_final, answer4_final = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Finalize unique rectangle set after debate.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4_final,
        "answer": answer4_final
    }
    logs.append(subtask_desc4)

    reflect_inst_5 = (
        "Sub-task 5: Aggregate the verified unique rectangles from Subtask 4 to produce the final count of rectangles formed inside the regular dodecagon with sides on polygon edges or diagonals. "
        "Provide a detailed summary of the counting method, including how the planar graph and intersection points were used. "
        "Perform a final verification by grouping rectangles according to direction pairs or symmetry classes and cross-check counts with alternative combinatorial or geometric arguments. "
        "Return the final numeric answer alongside verification results and a reflection on completeness and correctness."
    )
    cot_reflect_instruction_5 = "Sub-task 5: Aggregate and finalize the count of rectangles formed inside the dodecagon." + reflect_inst_5
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_5 = [taskInfo, thinking4_final, answer4_final]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion | CoT"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, aggregating and finalizing count, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5],
                                               "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining final count, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
