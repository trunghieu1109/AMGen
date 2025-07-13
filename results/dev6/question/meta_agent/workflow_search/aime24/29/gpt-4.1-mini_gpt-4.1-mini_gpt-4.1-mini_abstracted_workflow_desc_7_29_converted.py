async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent_base = LLMAgentBase
    N_sc = self.max_sc
    N_max = self.max_round

    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent the problem setting by defining variables for the color assignments of each of the 5 rows and 5 columns, and the occupancy of each cell. "
        "Clarify the interpretation of the conditions: (a) each cell contains at most one chip, (b) all chips in the same row have the same color, "
        "(c) all chips in the same column have the same color, and (d) maximality means no additional chip can be added without violating these conditions. "
        "Explicitly state assumptions about empty rows and columns, particularly that empty rows/columns correspond to no chips placed and thus no color assignment. "
        "This formalization must serve as the foundation for all subsequent reasoning and explicitly note that the color assignments are from {white, black, none} with 'none' indicating empty lines."
    )
    cot_agent_0_1 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formal representation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Analyze compatibility constraints between row and column color assignments. "
        "Deduce that for a cell to be occupied, the row and column colors must agree and be non-empty. Characterize the possible color patterns of rows and columns that allow chip placement. "
        "Explicitly incorporate the intersection condition that a colored row must intersect at least one column of the same color and vice versa, ensuring no colored line is empty. "
        "Clarify how empty rows and columns affect these constraints and occupancy. Refine the formal model by incorporating intersection and occupancy rules, preparing for maximality formalization."
    )
    cot_agents_0_2 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, compatibility constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo] + possible_answers_0_2 + possible_thinkings_0_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct solution for compatibility constraints.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Formulate the maximality condition precisely in terms of row and column color assignments and chip placements. "
        "Determine what it means for a configuration to be maximal: no empty cell can be filled without breaking uniformity conditions. "
        "Prove or rigorously justify that the sets of empty rows and empty columns must form a contiguous rectangular block (a submatrix) of empty cells. "
        "Enforce that every colored row (or column) intersects at least one column (or row) of the same color, ensuring no colored line is isolated. "
        "Explicitly record these constraints and assumptions in the shared context for all subsequent subtasks. This formalization is critical to avoid overcounting and invalid configurations in enumeration."
    )
    cot_agents_0_3 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, maximality condition, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3(
        [taskInfo] + possible_answers_0_3 + possible_thinkings_0_3,
        "Sub-task 3: Synthesize and choose the most consistent and correct solution for maximality condition.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    debate_instruction_0_4 = (
        "Sub-task 4: Verify the maximality formalization and rectangular block assumption by debating among agents. "
        "Use small grid cases (e.g., 1x1, 2x2) to validate or refute these assumptions, ensuring early detection of inconsistencies. "
        "Challenge and confirm the correctness of the maximality formalization before proceeding."
    )
    debate_agents_0_4 = [cot_agent_base(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": debate_instruction_0_4,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "Debate"
    }
    all_thinking_0_4 = []
    all_answer_0_4 = []
    for r in range(N_max):
        round_thinking = []
        round_answer = []
        for agent in debate_agents_0_4:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_3, answer_0_3], debate_instruction_0_4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking_0_3, answer_0_3] + all_thinking_0_4[-1] + all_answer_0_4[-1]
                thinking, answer = await agent(inputs, debate_instruction_0_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, maximality verification, thinking: {thinking.content}; answer: {answer.content}")
            round_thinking.append(thinking)
            round_answer.append(answer)
        all_thinking_0_4.append(round_thinking)
        all_answer_0_4.append(round_answer)
    final_decision_agent_0_4 = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo] + all_thinking_0_4[-1] + all_answer_0_4[-1],
                                                              "Sub-task 4: Final decision on maximality formalization correctness.",
                                                              is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Enumerate all possible assignments of colors to the 5 rows and 5 columns, where each line can be assigned white, black, or none (empty). "
        "Restrict the enumeration to assignments where the sets of empty rows and empty columns form contiguous intervals, i.e., rectangular blocks, as established in Stage 0. "
        "For each assignment, determine which cells can be occupied (cells where row and column colors match and are non-empty). "
        "Prepare this enumeration to respect all constraints derived in Stage 0, especially maximality and intersection conditions."
    )
    cot_agents_1_1 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_4, answer_0_4], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerate color assignments, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1: Synthesize and choose the most consistent enumeration of color assignments.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: For each candidate assignment from Sub-task 1, verify whether the resulting chip placement is maximal. "
        "Check that adding any chip to an empty cell would violate the uniformity conditions or the maximality constraints. "
        "This involves analyzing the structure of colored rows and columns, their intersections, and ensuring no colored line is empty or isolated. "
        "Discard non-maximal configurations. Explicitly filter out configurations violating the intersection or rectangular block conditions."
    )
    cot_agent_1_2 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1, thinking_0_3, answer_0_3], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, verify maximality, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    cot_sc_instruction_1_3a = (
        "Sub-task 3a: Count the number of distinct maximal configurations identified in Sub-task 2 by enumerating over all valid contiguous intervals of empty rows and columns and all valid color assignments to the colored rows and columns. "
        "Account for indistinguishability of chips within colors and the uniqueness of configurations under the problem constraints. "
        "Replace previous flawed combinatorial formulas with counts based on contiguous intervals and intersection constraints. "
        "Provide a closed-form or summation formula reflecting these refined counts."
    )
    cot_agents_1_3a = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_3a = {
        "subtask_id": "stage_1.subtask_3a",
        "instruction": cot_sc_instruction_1_3a,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_3a = []
    possible_thinkings_1_3a = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_3a[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_1_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3a[i].id}, count maximal configurations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3a.append(answer_i)
        possible_thinkings_1_3a.append(thinking_i)
    final_decision_agent_1_3a = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3a, answer_1_3a = await final_decision_agent_1_3a(
        [taskInfo] + possible_answers_1_3a + possible_thinkings_1_3a,
        "Sub-task 3a: Synthesize and choose the most consistent count of maximal configurations.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1.3a output: thinking - {thinking_1_3a.content}; answer - {answer_1_3a.content}")
    subtask_desc_1_3a['response'] = {"thinking": thinking_1_3a, "answer": answer_1_3a}
    logs.append(subtask_desc_1_3a)

    debate_instruction_1_3b = (
        "Sub-task 3b: Verify that no other maximal configurations exist outside the assumptions of rectangular blocks and intersection constraints. "
        "Use combinatorial arguments or small grid exhaustive checks to confirm completeness of the enumeration. "
        "This verification ensures that the counting formula from Sub-task 3a is exact and no configurations are missed or overcounted."
    )
    debate_agents_1_3b = [cot_agent_base(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    subtask_desc_1_3b = {
        "subtask_id": "stage_1.subtask_3b",
        "instruction": debate_instruction_1_3b,
        "context": ["user query", thinking_1_3a.content, answer_1_3a.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "Debate"
    }
    all_thinking_1_3b = []
    all_answer_1_3b = []
    for r in range(N_max):
        round_thinking = []
        round_answer = []
        for agent in debate_agents_1_3b:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_3a, answer_1_3a, thinking_0_4, answer_0_4], debate_instruction_1_3b, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking_1_3a, answer_1_3a, thinking_0_4, answer_0_4] + all_thinking_1_3b[-1] + all_answer_1_3b[-1]
                thinking, answer = await agent(inputs, debate_instruction_1_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verify completeness of maximal configurations, thinking: {thinking.content}; answer: {answer.content}")
            round_thinking.append(thinking)
            round_answer.append(answer)
        all_thinking_1_3b.append(round_thinking)
        all_answer_1_3b.append(round_answer)
    final_decision_agent_1_3b = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3b, answer_1_3b = await final_decision_agent_1_3b([taskInfo] + all_thinking_1_3b[-1] + all_answer_1_3b[-1],
                                                              "Sub-task 3b: Final decision on completeness and correctness of maximal configuration count.",
                                                              is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.3b output: thinking - {thinking_1_3b.content}; answer - {answer_1_3b.content}")
    subtask_desc_1_3b['response'] = {"thinking": thinking_1_3b, "answer": answer_1_3b}
    logs.append(subtask_desc_1_3b)

    reflect_inst_2_1 = (
        "Sub-task 1: Aggregate the counts from Stage 1 Sub-task 3a to derive the total number of valid maximal chip placements on the 5x5 grid. "
        "Confirm that the count respects all problem constraints and assumptions, including the rectangular block and intersection conditions. "
        "Prepare a final expression or numeric answer. This aggregation should explicitly reference the refined combinatorial structure and assumptions."
    )
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Your problem is to aggregate and confirm the final count of valid maximal chip placements. "
        + reflect_inst_2_1
    )
    cot_agent_2_1 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = cot_agent_base(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_1 = [taskInfo, thinking_1_3a, answer_1_3a]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_3a.content, answer_1_3a.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, aggregate counts, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1],
                                                 "Please review and provide limitations of the provided solution. If correct, output exactly 'True' in 'correct'.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining aggregate counts, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    reflect_inst_2_2 = (
        "Sub-task 2: Verify the correctness of the final count by cross-checking with alternative reasoning or smaller cases (e.g., 1x1, 2x2 grids). "
        "Reflect on the assumptions made and their impact on the solution. Provide a final answer with verification notes, highlighting consistency with the problem's maximality and structural constraints. "
        "If discrepancies arise, initiate a feedback loop to revisit earlier subtasks."
    )
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Your problem is to verify the correctness of the final count by alternative reasoning and smaller cases. "
        + reflect_inst_2_2
    )
    cot_agent_2_2 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion | CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_reflect_instruction_2_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verify correctness, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
