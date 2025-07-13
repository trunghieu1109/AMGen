async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Problem Formalization and Constraint Analysis

    cot_instruction_1 = (
        "Sub-task 1: Formally define the problem setting for the octagon coloring problem. "
        "Represent the octagon vertices as the set {0,...,7}, define the coloring as an 8-bit binary string where 0 = red and 1 = blue, "
        "and define the rotation group of order 8 acting on vertex indices by addition modulo 8. "
        "Clearly state the condition that there exists a rotation r such that the set of blue vertices after applying r is a subset of the original red vertices. "
        "Avoid any enumeration or probability calculation at this stage; focus solely on precise formalization of the problem elements and constraints."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing problem, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: For each rotation r in the group of order 8, analyze and characterize the constraints imposed on the coloring pattern by the condition that the blue vertices after rotation r map into originally red vertices. "
        "Express these constraints explicitly in terms of the binary string and the cycle decomposition of r. Treat each rotation separately and produce a clear description of the coloring restrictions per rotation."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, analyzing constraints per rotation, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2)
        possible_thinkings_0_2.append(thinking_0_2)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent answer for constraints per rotation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Cycle Structure and Independent Set Counting

    cot_sc_instruction_3 = (
        "Sub-task 3: For each rotation r, determine the cycle structure of r acting on the octagon vertices. "
        "Decompose the vertex set into disjoint cycles under r and express the coloring constraints on each cycle. "
        "Formulate the problem of counting valid colorings per cycle as counting independent sets (sets of vertices with no two adjacent blue vertices) on cycles of lengths dividing 8. "
        "Avoid using unverified formulas; instead, prepare to verify these counts explicitly."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, thinking_0_2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, determining cycle structures and constraints, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize and choose the most consistent answer for cycle structures and constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 3: ", sub_tasks[-1])

    cot_reflect_instruction_4 = (
        "Sub-task 4: Explicitly enumerate or derive the number of independent sets (colorings with no two adjacent blue vertices) on cycles of lengths 1, 2, 4, and 8. "
        "Produce small-case enumerations (e.g., list all independent sets for cycles of length 1, 2, 4) to verify the correctness of the counting formula. "
        "Confirm or correct the classical formula I(C_n) = F_{n-1} + F_{n+1} (with Fibonacci numbers defined as F_1=1, F_2=1). "
        "Document these enumerations and validations carefully to prevent propagation of errors. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_4 = self.max_round
    cot_inputs_1_4 = [taskInfo, thinking_1_3, answer_1_3]
    subtask_desc_1_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, enumerating independent sets and verifying formula, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    for i in range(N_max_1_4):
        feedback_1_4, correct_1_4 = await critic_agent_1_4([taskInfo, thinking_1_4], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_4.id}, providing feedback, thinking: {feedback_1_4.content}; answer: {correct_1_4.content}")
        if correct_1_4.content == "True":
            break
        cot_inputs_1_4.extend([thinking_1_4, feedback_1_4])
        thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, refining enumeration and verification, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 1.5: Brute-force Validation of Independent Set Counts

    reflexion_instruction_5 = (
        "Sub-task 5: Perform a brute-force enumeration of all 2^8 colorings of the octagon vertices to verify the counts of valid colorings per rotation obtained from the independent set formula. "
        "Compare the brute-force results with the formula-based counts for each rotation to validate correctness. "
        "If discrepancies arise, flag and halt further aggregation to avoid error propagation. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    reflexion_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_5 = self.max_round
    reflexion_inputs_1_5 = [taskInfo, thinking_1_4, answer_1_4]
    subtask_desc_1_5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_5,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "Reflexion | Debate"
    }
    thinking_1_5, answer_1_5 = await reflexion_agent_1_5(reflexion_inputs_1_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent_1_5.id}, brute-force validation of counts, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    for i in range(N_max_1_5):
        feedback_1_5, correct_1_5 = await critic_agent_1_5([taskInfo, thinking_1_5], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_5.id}, providing feedback, thinking: {feedback_1_5.content}; answer: {correct_1_5.content}")
        if correct_1_5.content == "True":
            break
        reflexion_inputs_1_5.extend([thinking_1_5, feedback_1_5])
        thinking_1_5, answer_1_5 = await reflexion_agent_1_5(reflexion_inputs_1_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {reflexion_agent_1_5.id}, refining brute-force validation, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    subtask_desc_1_5['response'] = {"thinking": thinking_1_5, "answer": answer_1_5}
    logs.append(subtask_desc_1_5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Inclusion-Exclusion Setup and Intersection Computations

    cot_instruction_6 = (
        "Sub-task 6: Define the sets A_r of colorings that satisfy the condition for each rotation r. "
        "Explicitly formulate the problem of finding the size of the union of these sets, i.e., colorings admitting at least one such rotation. "
        "Emphasize that Burnside's lemma counts orbits and is not applicable for union size. Prepare to apply inclusion-exclusion principle stepwise to compute the union size."
    )
    cot_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking_1_5.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_6, answer_2_6 = await cot_agent_2_6([taskInfo, thinking_1_5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_6.id}, defining sets and union problem, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_2_6.content}; answer - {answer_2_6.content}")
    subtask_desc_2_6['response'] = {"thinking": thinking_2_6, "answer": answer_2_6}
    logs.append(subtask_desc_2_6)
    print("Step 6: ", sub_tasks[-1])

    cot_sc_instruction_7 = (
        "Sub-task 7: Compute the sizes of all pairwise intersections A_r ∩ A_s for distinct rotations r and s. "
        "Use the cycle structures of the combined rotations and the independent set counts verified earlier to determine these intersection sizes accurately. "
        "Document the methodology and results clearly."
    )
    cot_agents_2_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2_7 = []
    possible_thinkings_2_7 = []
    subtask_desc_2_7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", thinking_2_6.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_7, answer_2_7 = await cot_agents_2_7[i]([taskInfo, thinking_2_6], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_7[i].id}, computing pairwise intersections, thinking: {thinking_2_7.content}; answer: {answer_2_7.content}")
        possible_answers_2_7.append(answer_2_7)
        possible_thinkings_2_7.append(thinking_2_7)
    final_decision_agent_2_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_7, answer_2_7 = await final_decision_agent_2_7([taskInfo] + possible_thinkings_2_7, "Sub-task 7: Synthesize and choose the most consistent answer for pairwise intersections.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_2_7.content}; answer - {answer_2_7.content}")
    subtask_desc_2_7['response'] = {"thinking": thinking_2_7, "answer": answer_2_7}
    logs.append(subtask_desc_2_7)
    print("Step 7: ", sub_tasks[-1])

    cot_sc_instruction_8 = (
        "Sub-task 8: Compute the sizes of all triple intersections A_r ∩ A_s ∩ A_t for distinct rotations r, s, t. "
        "Extend the methodology from pairwise intersections, carefully analyzing combined cycle structures and applying the independent set counts. "
        "Ensure correctness through cross-verification."
    )
    cot_agents_2_8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2_8 = []
    possible_thinkings_2_8 = []
    subtask_desc_2_8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_sc_instruction_8,
        "context": ["user query", thinking_2_7.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_8, answer_2_8 = await cot_agents_2_8[i]([taskInfo, thinking_2_7], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_8[i].id}, computing triple intersections, thinking: {thinking_2_8.content}; answer: {answer_2_8.content}")
        possible_answers_2_8.append(answer_2_8)
        possible_thinkings_2_8.append(thinking_2_8)
    final_decision_agent_2_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_8, answer_2_8 = await final_decision_agent_2_8([taskInfo] + possible_thinkings_2_8, "Sub-task 8: Synthesize and choose the most consistent answer for triple intersections.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking_2_8.content}; answer - {answer_2_8.content}")
    subtask_desc_2_8['response'] = {"thinking": thinking_2_8, "answer": answer_2_8}
    logs.append(subtask_desc_2_8)
    print("Step 8: ", sub_tasks[-1])

    cot_sc_instruction_9 = (
        "Sub-task 9: If necessary, compute higher-order intersections (quadruple and beyond) of the sets A_r to complete the inclusion-exclusion formula. "
        "Assess whether these are needed based on the group size and intersection patterns. Document all intersection sizes and their derivations."
    )
    cot_agents_2_9 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2_9 = []
    possible_thinkings_2_9 = []
    subtask_desc_2_9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_sc_instruction_9,
        "context": ["user query", thinking_2_8.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_9, answer_2_9 = await cot_agents_2_9[i]([taskInfo, thinking_2_8], cot_sc_instruction_9, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_9[i].id}, computing higher-order intersections, thinking: {thinking_2_9.content}; answer: {answer_2_9.content}")
        possible_answers_2_9.append(answer_2_9)
        possible_thinkings_2_9.append(thinking_2_9)
    final_decision_agent_2_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_9, answer_2_9 = await final_decision_agent_2_9([taskInfo] + possible_thinkings_2_9, "Sub-task 9: Synthesize and choose the most consistent answer for higher-order intersections.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking_2_9.content}; answer - {answer_2_9.content}")
    subtask_desc_2_9['response'] = {"thinking": thinking_2_9, "answer": answer_2_9}
    logs.append(subtask_desc_2_9)
    print("Step 9: ", sub_tasks[-1])

    debate_instruction_10 = (
        "Sub-task 10: Apply the inclusion-exclusion principle stepwise using the computed sizes of single sets, pairwise, triple, and higher-order intersections to find the exact size of the union of the sets A_r. "
        "Carefully verify each step to avoid double counting or omission. Cross-check the final count for consistency and plausibility. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_10 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_10 = self.max_round
    all_thinking_2_10 = [[] for _ in range(N_max_2_10)]
    all_answer_2_10 = [[] for _ in range(N_max_2_10)]
    subtask_desc_2_10 = {
        "subtask_id": "subtask_10",
        "instruction": debate_instruction_10,
        "context": ["user query", thinking_2_9.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_10):
        for i, agent in enumerate(debate_agents_2_10):
            if r == 0:
                thinking_2_10, answer_2_10 = await agent([taskInfo, thinking_2_9], debate_instruction_10, r, is_sub_task=True)
            else:
                input_infos_2_10 = [taskInfo, thinking_2_9] + all_thinking_2_10[r-1]
                thinking_2_10, answer_2_10 = await agent(input_infos_2_10, debate_instruction_10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, inclusion-exclusion aggregation, thinking: {thinking_2_10.content}; answer: {answer_2_10.content}")
            all_thinking_2_10[r].append(thinking_2_10)
            all_answer_2_10[r].append(answer_2_10)
    final_decision_agent_2_10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_10, answer_2_10 = await final_decision_agent_2_10([taskInfo] + all_thinking_2_10[-1], "Sub-task 10: Finalize inclusion-exclusion union size calculation.", is_sub_task=True)
    agents.append(f"Final Decision agent, inclusion-exclusion final aggregation, thinking: {thinking_2_10.content}; answer: {answer_2_10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking_2_10.content}; answer - {answer_2_10.content}")
    subtask_desc_2_10['response'] = {"thinking": thinking_2_10, "answer": answer_2_10}
    logs.append(subtask_desc_2_10)
    print("Step 10: ", sub_tasks[-1])

    # Stage 3: Probability Computation and Final Answer

    cot_reflect_instruction_11 = (
        "Sub-task 11: Compute the probability that a random coloring admits at least one rotation satisfying the condition by dividing the union size obtained from inclusion-exclusion by the total number of colorings (2^8). "
        "Simplify the resulting fraction to lowest terms, ensuring m and n are relatively prime positive integers. Avoid approximations; provide exact fraction. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_11 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_11 = self.max_round
    cot_inputs_3_11 = [taskInfo, thinking_2_10, answer_2_10]
    subtask_desc_3_11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_reflect_instruction_11,
        "context": ["user query", thinking_2_10.content, answer_2_10.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_3_11, answer_3_11 = await cot_agent_3_11(cot_inputs_3_11, cot_reflect_instruction_11, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_11.id}, computing probability and simplifying fraction, thinking: {thinking_3_11.content}; answer: {answer_3_11.content}")
    for i in range(N_max_3_11):
        feedback_3_11, correct_3_11 = await critic_agent_3_11([taskInfo, thinking_3_11], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_11.id}, providing feedback, thinking: {feedback_3_11.content}; answer: {correct_3_11.content}")
        if correct_3_11.content == "True":
            break
        cot_inputs_3_11.extend([thinking_3_11, feedback_3_11])
        thinking_3_11, answer_3_11 = await cot_agent_3_11(cot_inputs_3_11, cot_reflect_instruction_11, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_11.id}, refining probability computation, thinking: {thinking_3_11.content}; answer: {answer_3_11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking_3_11.content}; answer - {answer_3_11.content}")
    subtask_desc_3_11['response'] = {"thinking": thinking_3_11, "answer": answer_3_11}
    logs.append(subtask_desc_3_11)
    print("Step 11: ", sub_tasks[-1])

    reflexion_instruction_12 = (
        "Sub-task 12: Calculate and output the final answer m + n, where m/n is the simplified probability fraction. "
        "Verify the correctness of the simplification and final arithmetic. Include a final verification step to confirm consistency with earlier enumerations and counts. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    reflexion_agent_3_12 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_12 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_12 = self.max_round
    reflexion_inputs_3_12 = [taskInfo, thinking_3_11, answer_3_11]
    subtask_desc_3_12 = {
        "subtask_id": "subtask_12",
        "instruction": reflexion_instruction_12,
        "context": ["user query", thinking_3_11.content, answer_3_11.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_12, answer_3_12 = await reflexion_agent_3_12(reflexion_inputs_3_12, reflexion_instruction_12, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent_3_12.id}, final arithmetic and verification, thinking: {thinking_3_12.content}; answer: {answer_3_12.content}")
    for i in range(N_max_3_12):
        feedback_3_12, correct_3_12 = await critic_agent_3_12([taskInfo, thinking_3_12], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_12.id}, providing feedback, thinking: {feedback_3_12.content}; answer: {correct_3_12.content}")
        if correct_3_12.content == "True":
            break
        reflexion_inputs_3_12.extend([thinking_3_12, feedback_3_12])
        thinking_3_12, answer_3_12 = await reflexion_agent_3_12(reflexion_inputs_3_12, reflexion_instruction_12, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {reflexion_agent_3_12.id}, refining final answer, thinking: {thinking_3_12.content}; answer: {answer_3_12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking_3_12.content}; answer - {answer_3_12.content}")
    subtask_desc_3_12['response'] = {"thinking": thinking_3_12, "answer": answer_3_12}
    logs.append(subtask_desc_3_12)
    print("Step 12: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_12, answer_3_12, sub_tasks, agents)
    return final_answer, logs
