async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1: Clarify and formalize the problem constraints and assumptions with precision. "
        "Define what it means for each row and column to be monochromatic in the context of chip placement, explicitly stating that only rows and columns containing chips are subject to the color uniformity condition. "
        "Formalize the maximality condition as requiring that no additional chip can be placed in any empty cell without violating the monochromatic row or column condition. "
        "Avoid ambiguous interpretations by explicitly stating that rows or columns without chips have no color assigned and thus no uniformity constraint. "
        "This subtask sets the foundational assumptions and definitions to be used consistently in all subsequent analysis and enumeration steps."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, clarifying problem constraints, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Analyze the structural implications of the constraints on the grid based on the formalized assumptions from subtask_1. "
        "Determine how the color assignments to rows and columns must be compatible so that the intersection cells do not force contradictory colors. "
        "Specifically, deduce that for any cell containing a chip, the row and column colors must match, and that for each color (black or white), the subsets of rows and columns assigned that color must be either both empty or both non-empty to avoid contradictions. "
        "Analyze how maximality restricts the possible empty cells and color assignments, ensuring that no further chips can be added without violating uniformity. "
        "This subtask produces a precise characterization of valid row and column color subset patterns and their compatibility conditions."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, analyzing structural constraints, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Enumerate all valid assignments of colors to rows and columns that satisfy the compatibility and maximality conditions derived in stage_1. "
        "Generate all quadruples (bR, bC, wR, wC) where bR and wR are disjoint subsets partitioning the rows, and bC and wC are disjoint subsets partitioning the columns, "
        "with the conditions that for each color, either both the row and column subsets are empty or both are non-empty. "
        "For each such assignment, determine the resulting chip placement on the grid (cells where row and column colors agree). "
        "Explicitly enforce maximality by verifying that no additional chip can be placed in any empty cell without violating the uniformity constraints. "
        "Avoid double counting by treating black and white assignments jointly and ensuring disjointness and compatibility. "
        "Implement strict validation checks to filter out invalid or non-maximal configurations. "
        "This subtask integrates enumeration and maximality verification into a single, unified combinatorial counting process."
    )
    N_sc = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, enumerating valid color assignments with maximality, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)

    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2_1 = "Given all enumerations above, find the most consistent and correct count of maximal valid assignments of row and column colors satisfying compatibility and maximality."
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize consistent maximal valid assignments." + final_instr_2_1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate the counts of all maximal valid chip placements obtained from stage_2.subtask_1. "
        "Sum over all valid quadruple assignments that satisfy compatibility and maximality to produce the total number of ways to place chips on the grid under the problem's constraints. "
        "Provide a final numeric answer and verify its consistency with the problem conditions. "
        "Include a consistency check step: if any discrepancies or unexpected results arise, trigger a reflective subtask to reconcile and resolve conflicts before finalizing the answer."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_1, answer_2_1], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, aggregating final counts, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    reflect_inst = (
        "Sub-task 2: Perform a reflective verification of the aggregated result from subtask_1. "
        "Cross-check the final count against known constraints and logical bounds (e.g., maximality, color uniformity, chip count limits). "
        "If discrepancies or inconsistencies are detected, initiate a debate or reflection process to identify and correct errors in enumeration or aggregation. "
        "This subtask ensures the reliability and correctness of the final answer before reporting."
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_2 = {
        "subtask_id": "stage_3_subtask_2",
        "instruction": reflect_inst,
        "context": ["user query", thinking_3_1, answer_3_1],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2([taskInfo, thinking_3_1, answer_3_1], reflect_inst, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, verifying final count consistency, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
