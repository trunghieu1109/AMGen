async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0 Subtask 1: Mathematical characterization of b-eautiful integers (Reflexion + SC_CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formulate a precise mathematical characterization of b-eautiful integers. "
        "Express two-digit numbers in base b as n = x*b + y with digits constrained by 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "Derive the key equation x + y = sqrt(n) = sqrt(x*b + y). Analyze conditions under which n is a perfect square and the sum of digits equals its square root. "
        "Avoid assumptions about digits outside standard ranges or non-integer square roots. "
        "Goal: obtain a clear, usable form or parameterization of solutions for any given base b."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage0_subtask1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "Reflexion | SC_CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, mathematical characterization, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Stage 0 Subtask 2: Develop counting algorithm for fixed base b (CoT + SC_CoT)
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Develop a robust algorithm to test, for a fixed base b, all valid two-digit numbers (x,y) within digit constraints for the b-eautiful condition. "
        "Compute n = x*b + y, verify n is a perfect square, and check if x + y equals sqrt(n). "
        "Leverage algebraic insights from subtask 0.1 to optimize checks and handle edge cases. "
        "Ensure exhaustive and reliable enumeration without missing or double-counting solutions. "
        "Produce a reliable procedure for enumeration used in later counting."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage0_subtask2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, develop counting algorithm, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 0.2: Synthesize and finalize the counting algorithm for fixed base b.", is_sub_task=True)
    agents.append(f"Final Decision agent 0.2, finalize counting algorithm, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Stage 1 Subtask 3: Explicit enumeration of b-eautiful counts for bases 2 to 100 (CoT + SC_CoT)
    cot_sc_instruction_1_3 = (
        "Sub-task 3: Using the counting algorithm from stage 0.2, enumerate all b-eautiful integers for each base b from 2 to 100. "
        "Produce a complete numeric table mapping each base b to the exact count of b-eautiful integers. "
        "Ensure the enumeration is explicit, exhaustive, and verifiable. "
        "Output the table as a list or dictionary for further analysis."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage1_subtask3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, enumerate counts for bases 2-100, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Sub-task 1.3: Synthesize and finalize enumeration table for bases 2 to 100.", is_sub_task=True)
    agents.append(f"Final Decision agent 1.3, finalize enumeration table, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # Stage 1 Subtask 2 (modified): Identify minimal base b with count > 10 from enumeration table (Reflexion + SC_CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Analyze the enumeration table from Sub-task 1.3 to identify the minimal base b ≥ 2 for which the count of b-eautiful integers exceeds ten. "
        "This analysis must be purely data-driven, inspecting explicit counts rather than relying on heuristics. "
        "Clearly document the candidate base(s) where the threshold is first crossed."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion | SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, identify minimal base >10 count, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 1.2: Synthesize and finalize minimal base identification.", is_sub_task=True)
    agents.append(f"Final Decision agent 1.2, finalize minimal base identification, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Stage 1 Subtask 4: Cross-validation by brute force enumeration for small bases 2 to 20 (Debate + SC_CoT)
    debate_instruction_1_4 = (
        "Sub-task 4: Cross-validate the counting method by brute forcing the enumeration for small bases (2 to 20). "
        "Detect off-by-one errors, boundary issues, or missed solutions. "
        "Compare results with Sub-task 1.3 outputs to confirm accuracy and completeness. "
        "Report discrepancies and refine counting algorithm if necessary. "
        "This cross-validation ensures confidence in enumeration correctness before final verification."
    )
    debate_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_4 = self.max_round
    all_thinking_1_4 = [[] for _ in range(N_max_1_4)]
    all_answer_1_4 = [[] for _ in range(N_max_1_4)]
    subtask_desc_1_4 = {
        "subtask_id": "stage1_subtask4",
        "instruction": debate_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Debate | SC_CoT"
    }
    for r in range(N_max_1_4):
        for i, agent in enumerate(debate_agents_1_4):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_3, answer_1_3], debate_instruction_1_4, r, is_sub_task=True)
            else:
                input_infos_1_4 = [taskInfo, thinking_1_3, answer_1_3] + all_thinking_1_4[r-1] + all_answer_1_4[r-1]
                thinking_i, answer_i = await agent(input_infos_1_4, debate_instruction_1_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validation, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_1_4[r].append(thinking_i)
            all_answer_1_4[r].append(answer_i)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + all_thinking_1_4[-1] + all_answer_1_4[-1], "Sub-task 1.4: Finalize cross-validation results.", is_sub_task=True)
    agents.append(f"Final Decision agent 1.4, finalize cross-validation, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)

    # Stage 2 Subtask 1: Explicit enumeration and listing of all b-eautiful integers for candidate base(s) (Reflexion + CoT + Debate)
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: For the candidate base(s) identified in stage 1.2, explicitly enumerate and list all b-eautiful integers with their digit pairs and values. "
        "Re-run the counting algorithm solely for these bases to confirm correctness and completeness. "
        "Provide a detailed verification report including the list of all b-eautiful integers found, their digit decompositions, and square roots. "
        "Use Reflexion and Debate to ensure rigorous verification and prevent premature conclusions."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    subtask_desc_2_1 = {
        "subtask_id": "stage2_subtask1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion | CoT | Debate"
    }
    cot_inputs_2_1 = [taskInfo, thinking_1_2, answer_1_2]
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, enumerate and verify candidate bases, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback: {feedback_2_1.content}; correct: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining verification, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    for r in range(N_max_2_1):
        for j, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking_2_1, answer_2_1], cot_reflect_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_2_1, answer_2_1] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_d, answer_d = await agent(input_infos_2_1, cot_reflect_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verification debate, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_2_1[r].append(thinking_d)
            all_answer_2_1[r].append(answer_d)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1_final, answer_2_1_final = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 2.1: Final verification and listing of b-eautiful integers for candidate bases.", is_sub_task=True)
    agents.append(f"Final Decision agent 2.1, finalize verification, thinking: {thinking_2_1_final.content}; answer: {answer_2_1_final.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1_final.content}; answer - {answer_2_1_final.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1_final, "answer": answer_2_1_final}
    logs.append(subtask_desc_2_1)

    # Stage 2 Subtask 2: Synthesize final answer with verification report (Reflexion + SC_CoT)
    cot_sc_instruction_2_2 = (
        "Sub-task 2: Synthesize all findings to produce the final answer: the least integer base b ≥ 2 for which there are more than ten b-eautiful integers. "
        "Present the answer alongside the verification report from Sub-task 2.1 and discuss any edge cases or subtleties encountered. "
        "Ensure the conclusion is fully supported by explicit enumeration and verification data, eliminating heuristic or assumption-based errors. "
        "This final summary completes the workflow with a confident, data-backed solution."
    )
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage2_subtask2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1_final.content, answer_2_1_final.content],
        "agent_collaboration": "Reflexion | SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_2_2[i]([taskInfo, thinking_2_1_final, answer_2_1_final], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, synthesize final answer, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2.2: Final synthesis and answer.", is_sub_task=True)
    agents.append(f"Final Decision agent 2.2, final answer synthesis, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    print("Step 0.1: ", sub_tasks[-7])
    print("Step 0.2: ", sub_tasks[-6])
    print("Step 1.3: ", sub_tasks[-5])
    print("Step 1.2: ", sub_tasks[-4])
    print("Step 1.4: ", sub_tasks[-3])
    print("Step 2.1: ", sub_tasks[-2])
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
