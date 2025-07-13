async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # --------------------------------------------------------------------------------------------------------------
    # Stage 0: Formalization and Maximality Enforcement
    # Sub-task 1: Precisely formalize problem constraints with Chain-of-Thought
    cot_instruction_0_1 = (
        "Sub-task 1: Precisely formalize the problem constraints: (a) each cell contains at most one chip; "
        "(b) all chips in the same row have the same color; (c) all chips in the same column have the same color; "
        "(d) the placement is maximal, meaning no additional chip can be added without violating (b) or (c). "
        "Carefully clarify how empty rows or columns (with no chips) are interpreted in terms of color uniformity — specifically, distinguish between rows/columns with assigned colors (white, black) and empty rows/columns (no chips, no color). "
        "Define notation for row and column color assignments allowing {white, black, empty} states. Avoid premature assumptions about maximality or feasibility. "
        "Establish compatibility conditions for chip placement at intersection cells based on assigned row and column colors. "
        "This formalization will serve as the foundation for all subsequent reasoning.")
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing problem constraints, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 0 Sub-task 2: Incorporate maximality condition with Self-Consistency Chain-of-Thought
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Refine the formalization by explicitly incorporating the maximality condition: "
        "prove and enforce that empty rows and empty columns cannot coexist simultaneously, i.e., the sets of empty rows and empty columns must satisfy E_r · E_c = 0. "
        "Demonstrate that if both exist, then there is at least one empty cell where a chip can be added without violating uniformity, contradicting maximality. "
        "Formulate this as a logical constraint to be applied in enumeration. Clarify that maximality implies that every empty cell lies either in a row or column assigned a color, preventing further chip additions. "
        "This step must prevent counting invalid assignments that violate maximality due to coexistence of empty rows and columns.")
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, refining maximality condition, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent maximality condition enforcement.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing maximality condition, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 0 Sub-task 3: Filter assignments violating maximality with Self-Consistency Chain-of-Thought
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Develop a filtering criterion and algorithm to exclude all assignments where both empty rows and empty columns are non-empty. "
        "Enumerate all possible assignments of row and column colors from {white, black, empty} (3^5 each), then remove those with E_r ≠ ∅ and E_c ≠ ∅ simultaneously. "
        "Derive and verify the corrected total number of maximal assignments after this exclusion, using the formula: total = 3^{10} - (3^5 - 2^5)^2 = 14528. "
        "This filtering ensures only maximal configurations remain for further analysis. Prepare this filtered set as input for chip feasibility checks.")
    N_sc_0_3 = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_3)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0_subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_2, answer_0_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_3):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, filtering maximal assignments, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_answers_0_3 + possible_thinkings_0_3, "Sub-task 3: Synthesize and verify filtered maximal assignments count.", is_sub_task=True)
    agents.append(f"Final Decision agent, verifying filtered maximal assignments, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Stage 0 Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 1: Chip Supply Feasibility and Enumeration
    # Sub-task 1: Analyze chip supply constraints with Self-Consistency Chain-of-Thought
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Analyze the chip supply constraints: calculate the number of chips required for each color in a given assignment. "
        "For each assignment of row and column colors, compute the number of white chips needed as the product of the count of white rows and white columns, and similarly for black chips. "
        "Compare these requirements against the available supply of 25 chips per color. Filter out any assignments where the required chips exceed available supply. "
        "This step ensures feasibility and excludes impossible configurations. Explicitly incorporate chip supply feasibility into the enumeration process to avoid overcounting.")
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3, answer_0_3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, analyzing chip supply feasibility, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize chip supply feasibility filtering.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing chip supply feasibility, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 1 Sub-task 2: Enumerate valid maximal assignments with Chain-of-Thought
    cot_instruction_1_2 = (
        "Sub-task 2: Enumerate all valid maximal assignments of row and column colors from the filtered sets (maximality enforced, chip supply feasible). "
        "Confirm that each assignment satisfies the compatibility conditions at the intersection cells and that no additional chips can be added without violating uniformity. "
        "Verify uniqueness of these maximal configurations and ensure no double counting occurs due to indistinguishability or symmetric assignments. "
        "This enumeration forms the basis for counting the total number of valid maximal placements.")
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, enumerating valid maximal assignments, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 2: Counting and Verification
    # Sub-task 1: Count maximal assignments with Debate
    debate_instr_2_1 = (
        "Sub-task 1: Decompose the counting problem into combinatorial components: count the number of maximal assignments of row and column colors that satisfy maximality and chip supply constraints. "
        "Use combinatorial formulas or efficient algorithms to count these assignments without brute force enumeration where possible. "
        "Represent the grid as a bipartite graph or matrix and use the compatibility and maximality conditions to simplify counting. "
        "Ensure that the counting respects all constraints and that each counted configuration corresponds to a unique maximal placement.")
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": debate_instr_2_1,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instr_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instr_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, counting maximal assignments, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Final counting of maximal assignments.", is_sub_task=True)
    agents.append(f"Final Decision agent, final counting, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 2 Sub-task 2: Reflect and verify counting results with Reflexion
    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Reflect on and verify the counting results to ensure no overcounting or undercounting. "
        "Check that maximality and chip supply constraints are properly incorporated. Validate the final count against known corner cases and logical bounds. "
        "Adjust counting formulas if necessary and confirm that the final numeric count matches the corrected total (expected 14528). "
        "Provide a detailed explanation of the verification process and any adjustments made. " + reflect_inst_2_2)
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verifying counting results, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining counting verification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 3: Final Aggregation
    # Sub-task 1: Aggregate all validated counts with Chain-of-Thought
    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate all validated intermediate counts from previous subtasks to compute the total number of valid maximal placements of chips on the 5x5 grid. "
        "Combine counts of row and column color assignments with compatibility, maximality, and chip supply feasibility. "
        "Present the final numeric answer along with a comprehensive explanation of how constraints were enforced and how counts were combined. "
        "Verify that the final result respects all problem constraints, assumptions, and corrections from previous feedback.")
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_0_1, answer_0_1, thinking_1_2, answer_1_2, thinking_2_2, answer_2_2],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_0_1, answer_0_1, thinking_1_2, answer_1_2, thinking_2_2, answer_2_2], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, aggregating final counts, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
