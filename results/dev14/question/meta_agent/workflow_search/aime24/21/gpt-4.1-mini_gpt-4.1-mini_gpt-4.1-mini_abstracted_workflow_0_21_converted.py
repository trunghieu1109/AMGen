async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Polygon Geometry Formalization and Intersection Enumeration

    # Sub-task 1: Define geometric elements of the regular dodecagon
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the geometric elements of the regular dodecagon: "
        "12 vertices equally spaced on a circle, polygon sides as chords connecting adjacent vertices, "
        "and all diagonals as chords connecting non-adjacent vertices. Explicitly distinguish sides from diagonals."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining polygon elements, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Enumerate all chords (sides and diagonals) with endpoints, lengths, orientations
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Enumerate and represent all chords of the dodecagon, including sides and diagonals, "
        "verifying their existence and properties. Create a comprehensive data structure listing all chords with endpoints, lengths, and orientations. "
        "Allow rectangle vertices to be any intersection points of these chords, not just polygon vertices."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, enumerating chords, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 0.2: Synthesize and choose the most consistent chord enumeration.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Sub-task 3: Enumerate all intersection points of polygon sides and diagonals (including interior intersections)
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Enumerate all intersection points formed by the polygon's sides and diagonals, "
        "including polygon vertices and all interior intersection points where chords cross. "
        "Represent these points and their connectivity via chords in a graph or geometric data structure to serve as candidate vertices for rectangles."
    )
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, enumerating intersections, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 0.3: Synthesize and choose the most consistent intersection enumeration.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Sub-task 4: Precisely characterize rectangles allowing vertices at any intersection points
    debate_instruction_0_4 = (
        "Sub-task 4: Precisely characterize rectangles in this context by explicitly allowing rectangle vertices to be any intersection points of polygon sides or diagonals, not restricting to polygon vertices. "
        "Define rectangle properties: four vertices forming a convex quadrilateral with four right angles, and each side lying exactly on a polygon side or diagonal chord. Emphasize non-degeneracy and exact edge alignment with chords. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0_4 = self.max_round
    all_thinking_0_4 = [[] for _ in range(N_max_0_4)]
    all_answer_0_4 = [[] for _ in range(N_max_0_4)]
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": debate_instruction_0_4,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_4):
        for i, agent in enumerate(debate_agents_0_4):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_0_3], debate_instruction_0_4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_3] + all_thinking_0_4[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_0_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, characterizing rectangles, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_0_4[r].append(thinking_i)
            all_answer_0_4[r].append(answer_i)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo] + all_thinking_0_4[-1], "Sub-task 0.4: Final decision on rectangle characterization.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 0.4: ", sub_tasks[-1])

    # Sub-task 5: Formulate geometric constraints for rectangles
    cot_sc_instruction_0_5 = (
        "Sub-task 5: Formulate geometric constraints for rectangles in terms of the intersection points and chords: "
        "right angles via vector dot products, parallelism of opposite sides, and edge alignment with chords. "
        "Develop algebraic or geometric criteria to verify candidate rectangles formed from the intersection graph."
    )
    cot_agents_0_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_5 = []
    possible_thinkings_0_5 = []
    subtask_desc_0_5 = {
        "subtask_id": "stage_0.subtask_5",
        "instruction": cot_sc_instruction_0_5,
        "context": ["user query", thinking_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_5[i]([taskInfo, thinking_0_4], cot_sc_instruction_0_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_5[i].id}, formulating rectangle constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_5.append(answer_i)
        possible_thinkings_0_5.append(thinking_i)
    final_decision_agent_0_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_5, answer_0_5 = await final_decision_agent_0_5([taskInfo] + possible_thinkings_0_5, "Sub-task 0.5: Synthesize rectangle geometric constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.5 output: thinking - {thinking_0_5.content}; answer - {answer_0_5.content}")
    subtask_desc_0_5['response'] = {"thinking": thinking_0_5, "answer": answer_0_5}
    logs.append(subtask_desc_0_5)
    print("Step 0.5: ", sub_tasks[-1])

    # Sub-task 6: Analyze symmetry and combinatorial structure to reduce enumeration
    cot_sc_instruction_0_6 = (
        "Sub-task 6: Analyze the symmetry and combinatorial structure of the regular dodecagon to identify equivalence classes of chords and intersection points, "
        "simplifying enumeration by exploiting rotational and reflectional symmetries. Use this to reduce redundant counting in later stages."
    )
    cot_agents_0_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_6 = []
    possible_thinkings_0_6 = []
    subtask_desc_0_6 = {
        "subtask_id": "stage_0.subtask_6",
        "instruction": cot_sc_instruction_0_6,
        "context": ["user query", thinking_0_5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_6[i]([taskInfo, thinking_0_5], cot_sc_instruction_0_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_6[i].id}, analyzing symmetry, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_6.append(answer_i)
        possible_thinkings_0_6.append(thinking_i)
    final_decision_agent_0_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_6, answer_0_6 = await final_decision_agent_0_6([taskInfo] + possible_thinkings_0_6, "Sub-task 0.6: Synthesize symmetry analysis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.6 output: thinking - {thinking_0_6.content}; answer - {answer_0_6.content}")
    subtask_desc_0_6['response'] = {"thinking": thinking_0_6, "answer": answer_0_6}
    logs.append(subtask_desc_0_6)
    print("Step 0.6: ", sub_tasks[-1])

    # Stage 1: Enumerate Candidate Rectangles

    # Sub-task 1: Enumerate all pairs of parallel chords that can serve as opposite sides of rectangles
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Enumerate all pairs of parallel chords (polygon sides or diagonals) from the verified chord set that can serve as opposite sides of a rectangle, "
        "using the symmetry classes and geometric constraints from Stage 0. Include chords of all valid lengths and orientations."
    )
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_6.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_6], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating parallel chord pairs, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1.1: Synthesize parallel chord pairs.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Sub-task 2: For each pair of parallel chords, find all pairs of perpendicular chords forming other sides
    cot_sc_instruction_1_2 = (
        "Sub-task 2: For each pair of parallel chords identified, find all pairs of perpendicular chords that can form the other two sides of a rectangle, "
        "ensuring all four sides lie exactly on polygon sides or diagonals and endpoints are intersection points."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, thinking_0_5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, thinking_0_5], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, finding perpendicular chord pairs, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 1.2: Synthesize perpendicular chord pairs.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Sub-task 3: Aggregate chord pairs into candidate rectangles and verify geometric properties
    cot_sc_instruction_1_3 = (
        "Sub-task 3: Aggregate the chord pairs into candidate rectangles by verifying that the four vertices are distinct intersection points, "
        "the quadrilateral is convex, has four right angles, and all edges coincide with polygon sides or diagonals. Filter out invalid or degenerate candidates."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, verifying candidate rectangles, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 1.3: Synthesize verified rectangles.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Validate assumption about rectangle vertices using Debate
    debate_instruction_1_4 = (
        "Sub-task 4: Implement a validation subtask to challenge and verify the assumption that rectangle vertices can be interior intersection points, "
        "using example rectangles from the diagram and hypothetical counterexamples. Use Debate to explore and confirm the correctness of the expanded vertex set assumption. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_4 = self.max_round
    all_thinking_1_4 = [[] for _ in range(N_max_1_4)]
    all_answer_1_4 = [[] for _ in range(N_max_1_4)]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": debate_instruction_1_4,
        "context": ["user query", thinking_1_3.content, thinking_0_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_4):
        for i, agent in enumerate(debate_agents_1_4):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_3, thinking_0_4], debate_instruction_1_4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_3, thinking_0_4] + all_thinking_1_4[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_1_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating vertex assumption, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_1_4[r].append(thinking_i)
            all_answer_1_4[r].append(answer_i)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + all_thinking_1_4[-1], "Sub-task 1.4: Final decision on vertex assumption validation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Final Counting and Validation

    # Sub-task 1: Compute total number of distinct rectangles accounting for symmetries
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Compute the total number of distinct rectangles formed by the candidate chord sets, "
        "accounting for symmetries (rotations and reflections) to avoid double counting. Use the symmetry analysis from Stage 0."
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_1 = [taskInfo, thinking_1_4, thinking_1_3]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_4.content, thinking_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, computing total rectangles, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2_1([taskInfo, thinking_2_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, feedback])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining total count, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Sub-task 2: Validate computed count by cross-checking with known examples and sanity checks
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Validate the computed count by cross-checking with known example rectangles (such as those shown in the diagram) "
        "and by verifying consistency with geometric constraints. Perform sanity checks and reconcile any discrepancies, ensuring the final count is complete and correct."
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, validating final count, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review and provide the limitations of provided validation. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining validation, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
