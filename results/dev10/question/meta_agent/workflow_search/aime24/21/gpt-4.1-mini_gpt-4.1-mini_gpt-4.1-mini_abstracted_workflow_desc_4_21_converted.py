async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # --------------------------------------------------------------------------------------------------------------
    # Stage 0: Geometry Setup and Rectangle Conditions
    # Subtask 1: Define vertices and enumerate all chords with coordinates and directions (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the regular dodecagon's geometry by computing the coordinates of its 12 vertices "
        "placed on the unit circle at angles multiples of 30 degrees. Enumerate all polygon sides and all diagonals, "
        "recording endpoints, lengths, and directions modulo 180 degrees. Validate completeness by confirming total counts and symmetry. "
        "Provide explicit coordinate data for later computations."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, enumerating vertices and chords, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # --------------------------------------------------------------------------------------------------------------
    # Stage 1: Enumerate and Verify Rectangle Candidates
    # Subtask 1: Enumerate all chords by direction classes and list lines explicitly (SC-CoT)
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Using the vertices and chords data from stage_0.subtask_1, enumerate and classify all polygon chords "
        "(sides and diagonals) by their direction classes based on angles modulo 180 degrees. Group lines into direction classes "
        "(e.g., 0°, 30°, 60°, 90°, 120°, 150°) and list all lines with endpoints and coordinates explicitly. Ensure completeness and accuracy without assumptions."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating chords by direction classes, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent enumeration of chords by direction classes.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Subtask 2: Enumerate candidate rectangles by selecting pairs of parallel lines and verify geometrically (SC-CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: From the enumerated direction classes and lines in stage_1.subtask_1, select all pairs of parallel lines within each direction class. "
        "For each such pair, identify all perpendicular direction classes and enumerate all pairs of lines in those classes that are perpendicular. "
        "Form candidate quadruples of lines (two pairs of parallel lines perpendicular to each other) that could form rectangle sides. "
        "For each candidate quadruple, compute the four intersection points using coordinate geometry, verify that these points correspond to polygon vertices or valid intersections of polygon sides/diagonals, "
        "and check rectangle properties: four right angles, equal opposite sides, and non-degeneracy. Exclude invalid candidates and provide a complete list of verified rectangles with vertex coordinates."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, enumerating and verifying rectangle candidates, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent and verified list of rectangles.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # --------------------------------------------------------------------------------------------------------------
    # Stage 2: Categorize and Count Rectangles (Debate)
    debate_instruction_2_1 = (
        "Sub-task 1: Categorize the verified rectangles from stage_1.subtask_2 based on geometric features such as orientation, vertex selection patterns, and symmetry classes. "
        "Use these categories to simplify counting and avoid double counting. Compute partial counts for each category, ensuring mutual exclusivity. Cross-validate counts with geometric constraints and provide detailed summaries and raw counts for each category."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, categorizing and counting rectangles, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Final synthesis of categorized rectangle counts.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # --------------------------------------------------------------------------------------------------------------
    # Stage 3: Final Aggregation and Verification (Reflexion)
    reflect_instruction_3_1 = (
        "Sub-task 1: Aggregate the partial counts of rectangles from all categories obtained in stage_2.subtask_1 to obtain the total number of rectangles inside the regular dodecagon whose sides lie on polygon sides or diagonals. "
        "Perform a final verification by applying the counting and verification method to smaller regular polygons (e.g., square, hexagon, octagon) with known rectangle counts to confirm correctness. "
        "Present the final count alongside verification results and assumptions. Discuss limitations or potential edge cases, confirming exclusion of degenerate or invalid rectangles. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_1 = self.max_round
    cot_inputs_3_1 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": reflect_instruction_3_1,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs_3_1, reflect_instruction_3_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, aggregating and verifying final count, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    critic_inst_3_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_3_1):
        feedback_3_1, correct_3_1 = await critic_agent_3_1([taskInfo, thinking_3_1, answer_3_1], critic_inst_3_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1.id}, providing feedback, thinking: {feedback_3_1.content}; answer: {correct_3_1.content}")
        if correct_3_1.content == "True":
            break
        cot_inputs_3_1.extend([thinking_3_1, answer_3_1, feedback_3_1])
        thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs_3_1, reflect_instruction_3_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining final count, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
