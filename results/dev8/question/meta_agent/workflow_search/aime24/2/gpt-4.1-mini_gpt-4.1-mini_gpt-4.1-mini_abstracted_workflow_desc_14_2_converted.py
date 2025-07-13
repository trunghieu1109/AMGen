async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Identify and clearly define the problem domain elements: the set of all possible colorings "
        "of the octagon's 8 vertices (each vertex independently red or blue with probability 1/2), and the group of rotations (8 rotations by multiples of 45 degrees). "
        "Verify the interpretation of the problem's condition that there exists a rotation r such that all blue vertices map onto vertices originally colored red (i.e., B ∩ r(B) = ∅). "
        "Clarify assumptions such as independence, equal probability, and rotation group structure including the identity rotation."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining domain and condition, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, formally characterize the subsets of vertices colored blue (B) and red (R) for a given coloring. "
        "Express the problem condition that there exists a rotation r such that r(B) ⊆ R equivalently as B ∩ r(B) = ∅. "
        "Develop precise set-theoretic and algebraic expressions for this condition, considering the cyclic group action on the vertex set. "
        "Avoid heuristic or informal descriptions; ensure formal clarity."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, formalizing condition, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2)
        possible_thinkings_0_2.append(thinking_0_2)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent formalization of the condition", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_1_1 = (
        "Sub-task 1: Derive a rigorous formal representation of the problem condition suitable for combinatorial enumeration and probability calculation. "
        "Define, for each rotation r^k, the set A_k of colorings where B ∩ r^k(B) = ∅. "
        "Express the overall favorable set as the union of all A_k for k=0 to 7. "
        "Prepare the framework for applying inclusion-exclusion or Burnside's lemma by clearly defining these sets and their intersections. "
        "Avoid skipping steps or assuming known formulas without proof or verification."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, deriving formal representation, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Validate the formal representation by applying it to small, manageable examples (e.g., polygons with 4 or 6 vertices) and special colorings (all red, all blue, alternating colors). "
        "Confirm that the condition and set definitions correctly capture the problem's requirement and that the counting approach aligns with intuitive expectations. "
        "Document any discrepancies and refine the formalism accordingly. Avoid assuming correctness without testing."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, validating representation on examples, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent validation results", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Implement an explicit enumeration of all 2^8 = 256 colorings of the octagon's vertices. "
        "For each coloring, systematically check for the existence of a rotation r^k such that B ∩ r^k(B) = ∅. "
        "Count the exact number of favorable colorings. Record intermediate results for each rotation and coloring to support verification."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_2, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, enumerating colorings and checking rotations, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Develop a closed-form combinatorial counting approach: for each rotation r^k, derive the exact count |A_k| of colorings satisfying B ∩ r^k(B) = ∅ using correct combinatorial formulas. "
        "Explicitly compute all necessary intersections |A_i ∩ A_j|, triple intersections, etc., applying inclusion-exclusion step-by-step with intermediate results. "
        "Cross-validate these counts with the enumeration results from subtask 1. Avoid skipping intersection computations or relying on unverified formulas."
    )
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_2, answer_2_2 = await cot_agents_2_2[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, combinatorial counting and inclusion-exclusion, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
        possible_answers_2_2.append(answer_2_2)
        possible_thinkings_2_2.append(thinking_2_2)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize and confirm combinatorial counting results", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_2_3 = (
        "Sub-task 3: Apply Burnside's lemma explicitly to count the number of colorings for which there exists a rotation mapping blue vertices onto originally red vertices. "
        "For each group element (rotation), count the number of colorings fixed under the disjointness condition. Combine these counts to find the total number of favorable colorings. "
        "Compare and reconcile results with those from enumeration and inclusion-exclusion. Avoid relying on incomplete or heuristic Burnside applications."
    )
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_3 = {
        "subtask_id": "stage_2_subtask_3",
        "instruction": cot_instruction_2_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3([taskInfo, thinking_2_2, answer_2_2], cot_instruction_2_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_3.id}, applying Burnside's lemma, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Stage 2 Sub-task 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 7: ", sub_tasks[-1])

    cot_sc_instruction_2_4 = (
        "Sub-task 4: Validate the counting approach by applying the enumeration and combinatorial methods to smaller polygons (e.g., squares or hexagons) where full enumeration is trivial. "
        "Confirm that the methods produce consistent and correct results before finalizing the octagon count. Document validation results and refine methods if discrepancies arise. Avoid skipping this validation step to ensure reliability."
    )
    cot_agents_2_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_4 = []
    possible_thinkings_2_4 = []
    subtask_desc_2_4 = {
        "subtask_id": "stage_2_subtask_4",
        "instruction": cot_sc_instruction_2_4,
        "context": ["user query", thinking_2_3.content, answer_2_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_4, answer_2_4 = await cot_agents_2_4[i]([taskInfo, thinking_2_3, answer_2_3], cot_sc_instruction_2_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_4[i].id}, validating counting on smaller polygons, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
        possible_answers_2_4.append(answer_2_4)
        possible_thinkings_2_4.append(thinking_2_4)
    final_decision_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_4, answer_2_4 = await final_decision_agent_2_4([taskInfo] + possible_answers_2_4 + possible_thinkings_2_4, "Sub-task 4: Synthesize and confirm validation results", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking_2_4, "answer": answer_2_4}
    logs.append(subtask_desc_2_4)
    print("Step 8: ", sub_tasks[-1])

    debate_instr_3_1 = (
        "Sub-task 1: Compute the final probability as the ratio of the number of favorable colorings to the total number of colorings (2^8). "
        "Simplify the fraction m/n to lowest terms, ensuring m and n are relatively prime positive integers. Then compute and return the sum m + n as the final answer. "
        "Verify the final answer by cross-checking calculations, confirming the fraction is in simplest form, and ensuring the reasoning aligns with the problem statement and assumptions."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": debate_instr_3_1,
        "context": ["user query", thinking_2_4.content, answer_2_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_2_4, answer_2_4], debate_instr_3_1, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_2_4, answer_2_4] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instr_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying and verifying fraction, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 1: Finalize and verify the simplified fraction and compute m+n", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3_1)
    print("Step 9: ", sub_tasks[-1])

    cot_sc_instruction_3_2 = (
        "Sub-task 2: Verify the final answer by cross-checking all calculations, confirming the fraction is in simplest form, and ensuring the reasoning aligns with the problem statement and assumptions. "
        "Include a detailed review of the enumeration, combinatorial derivations, and probability calculation. Provide a final confirmation or identify any remaining issues. Avoid accepting the answer without thorough verification."
    )
    cot_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_2 = []
    possible_thinkings_3_2 = []
    subtask_desc_3_2 = {
        "subtask_id": "stage_3_subtask_2",
        "instruction": cot_sc_instruction_3_2,
        "context": ["user query", thinking_3.content, answer_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_3_2, answer_3_2 = await cot_agents_3_2[i]([taskInfo, thinking_3, answer_3], cot_sc_instruction_3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_2[i].id}, verifying final answer, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
        possible_answers_3_2.append(answer_3_2)
        possible_thinkings_3_2.append(thinking_3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + possible_answers_3_2 + possible_thinkings_3_2, "Sub-task 2: Synthesize and confirm final answer verification", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Sub-task 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
