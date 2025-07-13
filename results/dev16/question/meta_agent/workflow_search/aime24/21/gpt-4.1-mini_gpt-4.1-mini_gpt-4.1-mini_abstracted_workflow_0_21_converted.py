async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formal definitions and conditions

    # Subtask 1: Define geometric elements of the dodecagon (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the geometric elements of the regular dodecagon: its vertices, sides, and diagonals. "
        "Include precise definitions of vertex indexing (0 to 11), chord step sizes (1 to 6), and the properties of equal spacing on the circumscribed circle. "
        "Avoid enumeration or counting at this stage."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining dodecagon elements, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Characterize rectangle conditions (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Precisely characterize the necessary and sufficient conditions for a rectangle to be formed inside the dodecagon, "
        "focusing on the requirements that each rectangle side lies on either a side or a diagonal of the polygon, that rectangles have four right angles, "
        "and that opposite sides are parallel and equal in length. Emphasize that perpendicularity alone is insufficient without length equality."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, characterizing rectangle conditions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Subtask 3: Formalize polygon symmetry constraints (SC-CoT)
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Identify and formalize the constraints imposed by the polygon's symmetry and regularity on possible rectangle orientations and side alignments. "
        "Include the role of parallel chords, the necessity of pairs of parallel edges, and the implications of the polygon's rotational symmetry on chord directions and lengths."
    )
    N_sc = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_1, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, formalizing symmetry constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent answer for polygon symmetry constraints.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Subtask 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Subtask 4: Clarify assumptions and boundary conditions (Reflexion)
    reflect_inst_0_4 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_0_4 = (
        "Sub-task 4: Clarify all assumptions and boundary conditions relevant to the problem: confirm that all diagonals are considered, rectangles must be non-degenerate with positive area, "
        "and rectangles may lie fully inside or on the boundary of the polygon. Explicitly exclude degenerate cases and clarify the treatment of rectangles coinciding with polygon edges. "
        + reflect_inst_0_4
    )
    cot_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_0_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_0_4 = [taskInfo, thinking_0_1, thinking_0_2]
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": cot_reflect_instruction_0_4,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_4, answer_0_4 = await cot_agent_0_4(cot_inputs_0_4, cot_reflect_instruction_0_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_4.id}, clarifying assumptions, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    critic_inst_0_4 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback_0_4, correct_0_4 = await critic_agent_0_4([taskInfo, thinking_0_4], critic_inst_0_4, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_0_4.id}, providing feedback, thinking: {feedback_0_4.content}; answer: {correct_0_4.content}")
        if correct_0_4.content == "True":
            break
        cot_inputs_0_4.extend([thinking_0_4, feedback_0_4])
        thinking_0_4, answer_0_4 = await cot_agent_0_4(cot_inputs_0_4, cot_reflect_instruction_0_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_0_4.id}, refining assumptions, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    sub_tasks.append(f"Stage 0 Subtask 4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 0.4: ", sub_tasks[-1])

    # Stage 1: Enumerate and classify chords

    # Subtask 1: Enumerate and classify chords by step size and direction (CoT)
    cot_instruction_1_1 = (
        "Sub-task 1: Enumerate and classify all chords (sides and diagonals) of the dodecagon by their step sizes and directions. "
        "Compute and tabulate chord lengths for each step size from 1 to 6, and group chords into parallelism classes based on their directions to facilitate identification of potential rectangle sides."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_1.content, thinking_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_1, thinking_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, enumerating and classifying chords, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Generate candidate pairs of perpendicular chords (SC-CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Develop a method to generate all candidate pairs of perpendicular chords that can serve as adjacent sides of rectangles. "
        "Explicitly enumerate all valid (k,m) step-size pairs where m ≡ k + 3 mod 12 and m ≡ k - 3 mod 12, excluding invalid or degenerate cases. "
        "Emphasize that both perpendicular offsets must be considered to avoid undercounting."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, thinking_0_2], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, generating candidate perpendicular pairs, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent answer for candidate perpendicular chord pairs.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Subtask 3: Filter candidate pairs by chord length equality (Reflexion)
    reflect_inst_1_3 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_1_3 = (
        "Sub-task 3: Filter the candidate (k,m) pairs by enforcing chord length equality: retain only those pairs where chords of step sizes k and m have equal lengths, "
        "as required for opposite sides of rectangles. Use the chord length table computed previously to perform this filtering. "
        + reflect_inst_1_3
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_1_3 = [taskInfo, thinking_1_2, thinking_1_1]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2.content, thinking_1_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, filtering by chord length equality, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    critic_inst_1_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3], critic_inst_1_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback_1_3.content}; answer: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining chord length filtering, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Stage 1 Subtask 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Subtask 4: Transform problem into combinatorial counting (CoT)
    cot_instruction_1_4 = (
        "Sub-task 4: Transform the geometric problem into a combinatorial counting problem by representing rectangles as quadruples of vertices or chords satisfying the rectangle conditions derived previously. "
        "Express these conditions in terms of vertex indices and step sizes, incorporating the filtered (k,m) pairs and ensuring that all geometric constraints are encoded combinatorially."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_instruction_1_4,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, transforming problem combinatorially, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Stage 1 Subtask 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Enumeration, validation, and final counting

    # Subtask 1: Enumerate all valid rectangles (Reflexion)
    reflect_inst_2_1 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Enumerate all valid rectangles inside the dodecagon by systematically counting quadruples of vertices or chords that satisfy the rectangle conditions derived previously. "
        "Ensure that enumeration includes all valid (k,m) pairs from both perpendicular offsets and applies the chord length equality filter. "
        "Avoid explicit geometric construction but maintain rigorous combinatorial checks. "
        + reflect_inst_2_1
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_4], cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, enumerating valid rectangles, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    for i in range(self.max_round):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1 = [taskInfo, thinking_2_1, feedback_2_1]
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining enumeration, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Subtask 2: Validate enumeration results geometrically (Debate)
    debate_instr_2_2 = (
        "Sub-task 2: Validate the enumeration results by performing geometric verification of each candidate rectangle. "
        "Confirm that opposite sides correspond to chords of equal length, adjacent sides are perpendicular, and that rectangles are non-degenerate and lie inside or on the boundary of the polygon. "
        "Remove any invalid candidates identified during this validation. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_2 = self.max_round
    all_thinking_2_2 = [[] for _ in range(N_max_2_2)]
    all_answer_2_2 = [[] for _ in range(N_max_2_2)]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": debate_instr_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_2):
        for i, agent in enumerate(debate_agents_2_2):
            if r == 0:
                thinking_2_2, answer_2_2 = await agent([taskInfo, thinking_2_1], debate_instr_2_2, r, is_sub_task=True)
            else:
                input_infos_2_2 = [taskInfo, thinking_2_1] + all_thinking_2_2[r-1]
                thinking_2_2, answer_2_2 = await agent(input_infos_2_2, debate_instr_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating rectangles, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
            all_thinking_2_2[r].append(thinking_2_2)
            all_answer_2_2[r].append(answer_2_2)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + all_thinking_2_2[-1], "Sub-task 2: Final validation and filtering of rectangles.", is_sub_task=True)
    agents.append(f"Final Decision agent, validating enumeration results, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 Subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Subtask 3: Apply correct symmetry reduction (Reflexion)
    reflect_inst_2_3 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_2_3 = (
        "Sub-task 3: Apply the correct symmetry reduction by dividing the final count only by 4 to account for starting-vertex overcounting. "
        "Explicitly avoid any further division for dihedral symmetry, as rectangles in different positions are distinct. "
        "Confirm that the final count matches expected values based on the refined enumeration and validation. "
        + reflect_inst_2_3
    )
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_3 = [taskInfo, thinking_2_2]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_reflect_instruction_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, cot_reflect_instruction_2_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, applying symmetry reduction, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    critic_inst_2_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback_2_3, correct_2_3 = await critic_agent_2_3([taskInfo, thinking_2_3], critic_inst_2_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_3.id}, providing feedback, thinking: {feedback_2_3.content}; answer: {correct_2_3.content}")
        if correct_2_3.content == "True":
            break
        cot_inputs_2_3.extend([thinking_2_3, feedback_2_3])
        thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, cot_reflect_instruction_2_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, refining symmetry reduction, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Stage 2 Subtask 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
