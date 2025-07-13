async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Enumerate all chords of the regular dodecagon, including all sides and all diagonals (chords connecting non-adjacent vertices). "
        "For each chord, explicitly list its endpoints by vertex indices (0 to 11) and compute its geometric properties such as direction vector and slope. "
        "Produce a complete, explicit table of all chords with their endpoints and directions. Emphasize that no chords outside the polygon's vertices are considered. "
        "Avoid assumptions or omissions of any chord types. This explicit enumeration grounds all subsequent reasoning in concrete data."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Starting {subtask_desc_0_1['subtask_id']} with instruction: {cot_instruction_0_1}")
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, enumerating all chords, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Classify all enumerated chords from subtask_1 into distinct direction classes based on their geometric directions (e.g., slope or angle modulo 180 degrees). "
        "Since the dodecagon is regular, identify all unique directions and group chords accordingly. For each direction class, output the list of chords and the total count k_d. "
        "This classification is critical for identifying pairs of parallel lines and must be exhaustive and precise, including sides and diagonals. Avoid overlooking any direction class or misclassifying chords."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    print(f"Starting {subtask_desc_0_2['subtask_id']} with instruction: {cot_instruction_0_2}")
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, classifying chords into direction classes, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_instruction_0_3 = (
        "Sub-task 3: For each direction class identified in subtask_2, explicitly enumerate all pairs of parallel chords (lines) within that class. "
        "Compute the number of such pairs using the combinatorial formula choose(k_d, 2), where k_d is the number of chords in direction d. "
        "Output these counts and lists of pairs for all direction classes. This step replaces heuristic reasoning with explicit combinatorial enumeration, providing numeric data essential for rectangle counting. "
        "Avoid skipping this explicit counting or relying on symmetry alone."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Starting {subtask_desc_0_3['subtask_id']} with instruction: {cot_instruction_0_3}")
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, enumerating pairs of parallel chords, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    cot_instruction_1_1 = (
        "Sub-task 1: Identify all pairs of direction classes that are perpendicular to each other. "
        "For each such perpendicular pair (d, d_perp), use the counts of pairs of parallel chords from stage_0.subtask_3 to compute the number of candidate rectangles formed by these pairs as choose(k_d, 2) * choose(k_d_perp, 2). "
        "Explicitly list all such perpendicular direction pairs and their corresponding numeric rectangle counts. Emphasize that rectangles require two pairs of perpendicular parallel lines. "
        "Avoid heuristic or approximate counts; all calculations must be explicit and numeric."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    print(f"Starting {subtask_desc_1_1['subtask_id']} with instruction: {cot_instruction_1_1}")
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, computing rectangle counts from perpendicular direction pairs, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    debate_instruction_1_2 = (
        "Sub-task 2: Implement a programmatic enumeration that iterates over all vertex pairs and chords to explicitly identify and count all rectangles formed inside the dodecagon whose sides lie on polygon sides or diagonals. "
        "Verify that each candidate quadrilateral has four right angles, sides on chords, and is non-degenerate and inside or on the polygon boundary. "
        "Output the explicit list of rectangles found and their total count. This programmatic check serves as an independent verification of the combinatorial counts from subtask_1."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "Debate"
    }
    print(f"Starting {subtask_desc_1_2['subtask_id']} with instruction: {debate_instruction_1_2}")
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_1, answer_1_1], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_1, answer_1_1] + all_thinking_1_2[r-1] + all_answer_1_2[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating rectangles programmatically, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_1_2[r].append(thinking_i)
            all_answer_1_2[r].append(answer_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_2 = "Sub-task 2: Synthesize and choose the most consistent and correct programmatic enumeration of rectangles."
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + all_thinking_1_2[-1] + all_answer_1_2[-1], final_instr_1_2, is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing programmatic enumeration, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_1_3 = (
        "Sub-task 3: Cross-validate the rectangle counts obtained from the combinatorial calculations in subtask_1 and the programmatic enumeration in subtask_2. "
        "Identify and analyze any discrepancies, challenge assumptions, and reconcile differences through a Debate or multi-agent Reflexion pattern. "
        "Refine earlier subtasks if needed based on this verification. Provide a reconciled, verified numeric count of rectangles formed inside the dodecagon. "
        "Avoid accepting heuristic or unverified counts; ensure all results are consistent and justified."
    )
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_3 = self.max_round
    all_thinking_1_3 = [[] for _ in range(N_max_1_3)]
    all_answer_1_3 = [[] for _ in range(N_max_1_3)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instruction_1_3,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    print(f"Starting {subtask_desc_1_3['subtask_id']} with instruction: {debate_instruction_1_3}")
    for r in range(N_max_1_3):
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], debate_instruction_1_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2] + all_thinking_1_3[r-1] + all_answer_1_3[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validating rectangle counts, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_1_3[r].append(thinking_i)
            all_answer_1_3[r].append(answer_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_3 = "Sub-task 3: Synthesize and provide reconciled verified rectangle count after cross-validation."
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + all_thinking_1_3[-1] + all_answer_1_3[-1], final_instr_1_3, is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing reconciled count, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Aggregate the verified counts of all valid rectangles found in stage_1.subtask_3. "
        "Provide a final numeric answer for the total number of rectangles formed inside the regular dodecagon with sides on polygon sides or diagonals. "
        "Include a detailed justification referencing the explicit enumeration, combinatorial calculations, and programmatic verification. "
        "Ensure the final count aligns with geometric constraints and known examples (such as the three rectangles shown in the diagram). "
        "Avoid presenting the final answer without thorough verification."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Starting {subtask_desc_2_1['subtask_id']} with instruction: {cot_instruction_2_1}")
    for i in range(N_sc_2_1):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_3, answer_1_3], cot_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, aggregating verified rectangle counts, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2_1 = "Sub-task 1: Synthesize and choose the most consistent and correct total count of rectangles."
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, final_instr_2_1, is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing final rectangle count, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflect_inst_2_2 = (
        "Given previous subtasks and the aggregated count, carefully consider potential errors or oversights. "
        "Cross-check the count against known properties of regular polygons, symmetry arguments, and combinatorial formulas. "
        "If discrepancies arise, revisit earlier subtasks for refinement. Provide a final confirmed answer with justification."
    )
    cot_reflect_instruction_2_2 = "Sub-task 2: Verification and reflection on the total rectangle count." + reflect_inst_2_2
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Starting {subtask_desc_2_2['subtask_id']} with instruction: {cot_reflect_instruction_2_2}")
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verifying final count, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining verification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
