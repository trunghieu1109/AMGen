async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Identify and list all given edge lengths of tetrahedron ABCD, explicitly noting the pairs of equal edges and their exact radical values. Emphasize the geometric constraints and symmetry properties implied by these equalities without making assumptions about the tetrahedron's shape or orientation."
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, identify and list edge lengths, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1, "Sub-task 1: Synthesize and choose the most consistent answer for edge lengths and symmetry properties.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to analyze the implications of the given equal edge pairs on the tetrahedron's structure, focusing on identifying any symmetries or special properties that can simplify subsequent calculations. Avoid speculative assumptions and base conclusions strictly on the given data."
    cot_reflect_instruction_2 = "Sub-task 2: Analyze the implications of the given equal edge pairs on the tetrahedron's structure." + reflect_inst_2
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2 = [taskInfo, thinking1]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", "thinking of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, analyze edge pair implications, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2([taskInfo, thinking2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2.extend([thinking2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining analysis, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = "Sub-task 3: Compute the exact symbolic expressions for the areas of all four triangular faces of tetrahedron ABCD using Heron's formula. Express each area in simplest radical form, explicitly avoiding any decimal or numeric approximations. Document all algebraic steps and ensure radicals are simplified and exact."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content, thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, thinking2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, compute exact symbolic face areas, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = "Sub-task 4: Calculate the exact symbolic expression for the volume of tetrahedron ABCD using the Cayley-Menger determinant or an equivalent exact formula. Maintain all expressions in radical form without numeric approximation. Verify that the volume is positive and consistent with the tetrahedron's geometry."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking1.content, thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, thinking2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, compute exact symbolic volume, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflect_inst_5 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, verify the correctness and exactness of the symbolic expressions for the face areas and volume obtained in subtasks 3 and 4. This verification should include checking algebraic simplifications, radical forms, and consistency with geometric constraints. Flag any numeric approximations or inconsistencies for correction before proceeding."
    cot_reflect_instruction_5 = "Sub-task 5: Verify symbolic expressions for face areas and volume." + reflect_inst_5
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5 = [taskInfo, thinking3, thinking4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", thinking3.content, thinking4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verify symbolic expressions, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_5([taskInfo, thinking5], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = "Sub-task 6: Derive the exact symbolic expression for the inradius of tetrahedron ABCD using the formula inradius = (3 × volume) / (sum of face areas). Carry through all symbolic expressions from previous subtasks without numeric approximation. Rationalize denominators and simplify radicals algebraically as much as possible, but do not finalize the simplification into the form (m√n)/p yet."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking3.content, thinking4.content, thinking5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, thinking4, thinking5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, derive symbolic inradius expression, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    reflect_inst_7 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, perform a rigorous symbolic simplification and verification of the inradius expression derived in subtask 6. This includes simplifying radicals to ensure n is square-free, ensuring m and p are positive integers with no common factors, rationalizing denominators fully, and formally proving the expression matches the form (m√n)/p. Use step-by-step algebraic reasoning or computer algebra tools to avoid heuristic or numeric shortcuts."
    cot_reflect_instruction_7 = "Sub-task 7: Rigorous symbolic simplification and verification of inradius expression." + reflect_inst_7
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_7 = [taskInfo, thinking6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_7 = []
    possible_thinkings_7 = []
    for i in range(self.max_sc):
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_7.id}, symbolic simplification of inradius, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + possible_thinkings_7, "Sub-task 7: Synthesize and choose the most consistent and correct simplified inradius expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    reflect_inst_8 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, from the fully simplified and verified inradius expression in the form (m√n)/p, extract the integers m, n, and p. Confirm that m and p are coprime and n is square-free. Then compute and report the sum m + n + p as required by the problem. Provide a formal justification for the correctness of these values and the final sum."
    cot_reflect_instruction_8 = "Sub-task 8: Extract m, n, p and compute m+n+p with formal justification." + reflect_inst_8
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_8 = [taskInfo, thinking7]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, extract m,n,p and compute sum, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_8([taskInfo, thinking8], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining extraction and sum, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
