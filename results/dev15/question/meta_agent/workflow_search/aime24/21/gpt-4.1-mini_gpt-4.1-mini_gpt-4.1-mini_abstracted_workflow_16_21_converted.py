async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Identify and enumerate all chords (line segments) formed by the sides and all diagonals of the regular dodecagon, "
        "representing each chord by its endpoint vertex indices, direction (angle modulo 360 degrees), and length. "
        "Ensure wrap-around indices are normalized modulo 12 to avoid duplicates. Avoid any rectangle-related reasoning at this stage.")
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, enumerating chords, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Summarize the geometric properties of the regular dodecagon relevant to rectangle formation, "
        "including vertex arrangement on the unit circle, chord directions, and the polygon's symmetry group (rotations and reflections). "
        "Emphasize how these properties constrain possible rectangle sides and their orientations.")
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, summarizing geometric properties, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    reflect_inst_0_3 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better.")
    cot_reflect_instruction_0_3 = (
        "Sub-task 3: Clarify and explicitly state assumptions about which diagonals are included (all diagonals connecting any two distinct vertices), "
        "the nature of rectangles (non-degenerate, strictly inside or on polygon edges), and the criteria for distinctness of rectangles (distinct vertex sets modulo symmetry). "
        "Avoid ambiguity in these definitions. " + reflect_inst_0_3)
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_0_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_0_3 = self.max_round
    cot_inputs_0_3 = [taskInfo, thinking_0_1, thinking_0_2]
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_reflect_instruction_0_3,
        "context": ["user query", "thinking of stage_0.subtask_1", "thinking of stage_0.subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3(cot_inputs_0_3, cot_reflect_instruction_0_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_3.id}, clarifying assumptions, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    for i in range(N_max_0_3):
        feedback_0_3, correct_0_3 = await critic_agent_0_3([taskInfo, thinking_0_3],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_0_3.id}, providing feedback, thinking: {feedback_0_3.content}; answer: {correct_0_3.content}")
        if correct_0_3.content == "True":
            break
        cot_inputs_0_3.extend([thinking_0_3, feedback_0_3])
        thinking_0_3, answer_0_3 = await cot_agent_0_3(cot_inputs_0_3, cot_reflect_instruction_0_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_0_3.id}, refining assumptions, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Formally represent all chords from stage_0.subtask_1 as vectors with direction and length, "
        "and identify all pairs of parallel chords by comparing their directions modulo 180 degrees. "
        "Normalize directions to handle wrap-around and ensure no duplicates. Do not yet check perpendicularity or rectangle formation.")
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, identifying parallel chords, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent representation of parallel chord pairs." , is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Filter the parallel chord pairs identified in subtask_1 to retain only those pairs whose directions differ by exactly 90 degrees (i.e., chord directions differ by 3 steps around the 12-gon, corresponding to perpendicularity). "
        "Explicitly handle wrap-around indices and direction normalization.")
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1, thinking_0_2], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, filtering perpendicular chord pairs, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent filtered perpendicular chord pairs." , is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Establish the necessary geometric constraints for four chords to form a rectangle inside the dodecagon, including: adjacent edges are perpendicular, opposite edges are parallel and equal in length, and vertices are distinct and ordered to form a closed quadrilateral. "
        "Formalize these constraints using vector dot products and length equalities.")
    N_sc_1_3 = self.max_sc
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_3)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_0_2.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_3):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, thinking_0_2, answer_0_3], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, formalizing rectangle constraints, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize and choose the most consistent geometric constraints for rectangles." , is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_1_4 = (
        "Sub-task 4: Develop a combinatorial framework to generate candidate quadruples of vertices or chords that could form rectangles, "
        "using the filtered parallel and perpendicular chord pairs from subtask_2 and the geometric constraints from subtask_3. "
        "Avoid degenerate or invalid configurations by enforcing vertex distinctness and polygon containment.")
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_instruction_1_4,
        "context": ["user query", thinking_1_2.content, thinking_1_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_2, thinking_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, generating candidate quadruples, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Enumerate all candidate rectangles by checking each quadruple generated in stage_1.subtask_4 against the full geometric constraints: "
        "verify that adjacent edges are perpendicular (dot product zero within tolerance), opposite edges are equal in length, and the quadrilateral is simple and convex. "
        "Use explicit vector computations and avoid relying solely on length-based heuristics. Normalize vertex indices modulo 12 to handle wrap-around.")
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking_2_1, answer_2_1 = await cot_agents_2_1[i]([taskInfo, thinking_1_4], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, enumerating rectangles, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_thinkings_2_1, "Sub-task 1: Synthesize and choose the most consistent enumeration of candidate rectangles." , is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Analyze the symmetry group of the dodecagon (rotations by multiples of 30 degrees and reflections) and apply these symmetries to identify and eliminate duplicate rectangles from the enumeration in subtask_1. "
        "Normalize each rectangle's vertex set under these symmetries to ensure each distinct rectangle is counted exactly once.")
    N_sc_2_2 = self.max_sc
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_2)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_2):
        thinking_2_2, answer_2_2 = await cot_agents_2_2[i]([taskInfo, thinking_2_1, thinking_0_2], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, deduplicating rectangles by symmetry, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
        possible_answers_2_2.append(answer_2_2)
        possible_thinkings_2_2.append(thinking_2_2)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_thinkings_2_2, "Sub-task 2: Synthesize and choose the most consistent deduplicated rectangle set." , is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    debate_instr_2_3 = (
        "Sub-task 3: Verify the final set of candidate rectangles by re-checking all geometric conditions and ensuring no duplicates remain. "
        "Cross-validate the count with known theoretical results or symmetry arguments. Prepare a detailed report of the verification process and any assumptions made. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_3 = self.max_round
    all_thinking_2_3 = [[] for _ in range(N_max_2_3)]
    all_answer_2_3 = [[] for _ in range(N_max_2_3)]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": debate_instr_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_3):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking_2_3, answer_2_3 = await agent([taskInfo, thinking_2_2], debate_instr_2_3, r, is_sub_task=True)
            else:
                input_infos_2_3 = [taskInfo, thinking_2_2] + all_thinking_2_3[r-1]
                thinking_2_3, answer_2_3 = await agent(input_infos_2_3, debate_instr_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying rectangles, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
            all_thinking_2_3[r].append(thinking_2_3)
            all_answer_2_3[r].append(answer_2_3)
    final_decision_instr_2_3 = "Given all the above thinking and answers, reason over them carefully and provide a final verified answer."
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3([taskInfo] + all_thinking_2_3[-1], "Sub-task 3: Verification and reconciliation." + final_decision_instr_2_3, is_sub_task=True)
    agents.append(f"Final Decision agent, verifying rectangles, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    debate_instr_2_4 = (
        "Sub-task 4: Make the final decision on the total number of distinct rectangles formed inside the regular dodecagon with sides on polygon sides or diagonals. "
        "Justify the count by referencing the enumeration, symmetry considerations, and verification results. "
        "Address any discrepancies or conflicting counts from previous attempts and provide a clear, logically consistent conclusion. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_2_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_4 = self.max_round
    all_thinking_2_4 = [[] for _ in range(N_max_2_4)]
    all_answer_2_4 = [[] for _ in range(N_max_2_4)]
    subtask_desc_2_4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": debate_instr_2_4,
        "context": ["user query", thinking_2_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_4):
        for i, agent in enumerate(debate_agents_2_4):
            if r == 0:
                thinking_2_4, answer_2_4 = await agent([taskInfo, thinking_2_3], debate_instr_2_4, r, is_sub_task=True)
            else:
                input_infos_2_4 = [taskInfo, thinking_2_3] + all_thinking_2_4[r-1]
                thinking_2_4, answer_2_4 = await agent(input_infos_2_4, debate_instr_2_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, final decision on rectangle count, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
            all_thinking_2_4[r].append(thinking_2_4)
            all_answer_2_4[r].append(answer_2_4)
    final_decision_instr_2_4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_4, answer_2_4 = await final_decision_agent_2_4([taskInfo] + all_thinking_2_4[-1], "Sub-task 4: Final decision on total rectangles." + final_decision_instr_2_4, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing rectangle count, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking_2_4, "answer": answer_2_4}
    logs.append(subtask_desc_2_4)
    print("Step 2.4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_4, answer_2_4, sub_tasks, agents)
    return final_answer, logs
