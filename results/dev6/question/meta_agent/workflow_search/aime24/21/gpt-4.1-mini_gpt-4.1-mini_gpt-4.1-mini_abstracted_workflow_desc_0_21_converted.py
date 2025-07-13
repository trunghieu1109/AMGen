async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = (
        "Sub-task 1: Identify and list all relevant line segments in the regular dodecagon that can serve as sides of rectangles. "
        "Include all polygon sides and all diagonals connecting any two distinct vertices. Represent each segment precisely using vertex coordinates on the unit circle as 2D vectors or complex numbers. "
        "Output a structured JSON list of all 66 segments (12 sides + 54 diagonals) with endpoints and vector representations. Avoid excluding any diagonals unless explicitly specified. "
        "This structured data will be used in subsequent subtasks for geometric verification and enumeration."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc1}")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying line segments, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze and formalize the geometric conditions required for four segments to form a rectangle inside the dodecagon. "
        "This includes: (a) the quadrilateral formed by four vertices must have four right angles, (b) opposite sides must be equal and parallel, and (c) each side must lie on one of the identified polygon sides or diagonals. "
        "Use vector methods and exact symbolic computations to define criteria for perpendicularity and parallelism between segments. Explicitly state assumptions such as positive area and non-degeneracy. "
        "Output these conditions in a precise, machine-verifiable form (e.g., formulas or algorithms) to be used in enumeration and filtering."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before agent call: {subtask_desc2}")
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing rectangle conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent and correct geometric conditions for rectangles.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Enumerate all pairs of chord-length indices (k,m) modulo 12 that satisfy the perpendicularity condition for rectangle sides in a regular dodecagon. "
        "Specifically, find all pairs (k,m) with 1 ≤ k,m ≤ 5 such that k + m = 6, reflecting the polygon's symmetry and right angle conditions. "
        "Output these pairs as a structured JSON array. This subtask isolates the combinatorial geometric constraints on segment lengths and orientations, enabling systematic rectangle enumeration in the next subtask."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc3}")
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, enumerating chord-length pairs, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_4 = (
        "Sub-task 4: Using the chord-length pairs (k,m) from subtask_3 and the segment data from subtask_1, systematically enumerate all quadruples of distinct vertices (not restricted to cyclic order) that can form rectangles with sides on polygon sides or diagonals. "
        "For each candidate quadruple, verify rectangle conditions from subtask_2 using exact vector computations. Implement canonical ordering or hashing of vertex sets to avoid duplicate counting due to symmetry (rotations/reflections). "
        "Output a structured JSON list of all valid rectangles, each represented by its four vertices and corresponding side segments. This enumeration must be exhaustive and programmatically verifiable."
    )
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking2.content, answer2.content, thinking3.content, answer3.content, thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before agent call: {subtask_desc4}")
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking2, answer2, thinking3, answer3, thinking1, answer1], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking2, answer2, thinking3, answer3, thinking1, answer1] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating rectangles, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Synthesize and finalize the exhaustive enumeration of rectangles.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflexion_instruction_5 = (
        "Sub-task 5: Filter the enumerated rectangles from subtask_4 to retain only those fully contained inside the polygon and with positive area. "
        "Implement rigorous polygon containment tests (e.g., ray casting or winding number algorithms) to ensure no rectangle extends outside the dodecagon boundary. "
        "Confirm that all rectangle vertices are polygon vertices and that the rectangle is non-degenerate. Remove duplicates arising from symmetry not caught in subtask_4. "
        "Output the filtered list of valid rectangles in structured JSON format. This step ensures geometric validity and correctness of the final count."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_5,
        "context": ["user query", thinking4.content, answer4.content, thinking1.content, answer1.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before agent call: {subtask_desc5}")
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4, thinking1, answer1]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, filtering valid rectangles, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Please review and provide the limitations of provided solutions. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining filtered rectangles, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = (
        "Sub-task 6: Aggregate the count of all valid rectangles identified in subtask_5. "
        "Sum the total number of distinct rectangles formed inside the regular dodecagon with sides on polygon sides or diagonals. "
        "Cross-validate the count using known symmetry arguments and mathematical properties of the dodecagon to ensure no rectangles are missed or double-counted. "
        "Provide a final numeric answer alongside a verification report detailing the validation methods and results. Output the final count and verification in a structured format for transparency and reproducibility."
    )
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before agent call: {subtask_desc6}")
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    for agent in cot_agents_6:
        thinking6, answer6 = await agent([taskInfo, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT agent {agent.id}, aggregating final count, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_answers_6 + possible_thinkings_6, "Sub-task 6: Provide the final count of rectangles inside the dodecagon.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
