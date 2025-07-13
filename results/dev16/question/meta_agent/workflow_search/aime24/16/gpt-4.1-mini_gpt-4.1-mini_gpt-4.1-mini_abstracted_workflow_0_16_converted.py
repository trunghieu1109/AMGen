async def forward_16(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]

    reflexion_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    # Stage 1

    # Subtask 1: Formal representation of points O, I, A with O at origin, OA=13, IA ⟂ OI
    cot_instruction_1 = (
        "Sub-task 1: Formally represent the positions of points O (circumcenter), I (incenter), and A (vertex) in a coordinate system. "
        "Place O at the origin to simplify calculations. Express the given conditions OA = 13 (circumradius) and the perpendicularity condition IA ⟂ OI in vector or coordinate form. "
        "Avoid making any assumptions about the triangle's orientation or symmetry beyond what is necessary for simplification."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formal representation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Express incenter I in terms of triangle elements and relate to inradius r=6
    cot_instruction_2 = (
        "Sub-task 2: Express the incenter I in terms of the triangle's elements and relate its position to the inradius r = 6. "
        "Use known properties of the incenter and incircle to represent I's coordinates or vector position relative to O and A. "
        "Avoid assuming explicit side lengths or angles at this stage."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, express incenter I, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Translate IA ⟂ OI into algebraic equation linking I, A, O
    cot_sc_instruction_3 = (
        "Sub-task 3: Translate the perpendicularity condition IA ⟂ OI into an explicit algebraic equation involving the coordinates or vectors of points I, A, and O. "
        "Derive a constraint equation linking the unknown geometric parameters of the triangle. Focus on obtaining a usable relation without prematurely solving for side lengths or angles."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content, thinking2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3 = []
    possible_thinkings_3 = []
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_sc_agents[i]([taskInfo, thinking1, thinking2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, translate perpendicularity, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_final, answer3_final = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent algebraic relation for IA ⟂ OI.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize perpendicularity relation, thinking: {thinking3_final.content}; answer: {answer3_final.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    subtask_desc3['response'] = {"thinking": thinking3_final, "answer": answer3_final}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 4: Derive exact formula for cos A using IA = r / sin(A/2) and Pythagorean relation IA^2 + OI^2 = OA^2
    cot_instruction_4 = (
        "Sub-task 4: Derive an exact formula for cos A using the relation IA = r / sin(A/2) and the Pythagorean theorem applied to triangle O-I-A, given OA = R = 13 and OI = R - r = 7. "
        "Use the perpendicularity condition IA ⟂ OI to establish the relation IA² + OI² = OA². "
        "Avoid any assumptions about triangle symmetry or angle equality. Present the derivation rigorously and symbolically, resulting in cos A = (R - r) / R."
    )
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3_final.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent([taskInfo, thinking3_final], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, derive cos A formula, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Subtask 5: Express sin B sin C in terms of cos A and cos(B - C) without assuming B=C
    cot_instruction_5 = (
        "Sub-task 5: Express sin B sin C in terms of cos A and cos(B - C) using the identity sin B sin C = [cos(B - C) + cos A]/2. "
        "Do not assume B = C or any symmetry without proof. Keep cos(B - C) as a general variable to maintain full generality. "
        "Prepare this expression for use in subsequent calculations of AB · AC."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent([taskInfo, thinking4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, express sin B sin C, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Subtask 6: Verify correctness and consistency of cos A formula and sin B sin C expression
    reflect_inst_6 = (
        "Sub-task 6: Verify the correctness and consistency of the derived formula for cos A and the expression for sin B sin C. "
        "Critically evaluate the assumptions made so far, ensuring no unjustified simplifications such as isosceles assumptions have been introduced. "
        "Confirm that the perpendicularity condition and the incenter-incircle relations are fully respected. "
        "This verification step should prevent propagation of errors into the final computations."
    )
    cot_reflect_instruction_6 = "Sub-task 6: Your problem is to verify intermediate results." + " Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", thinking4.content, thinking5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await reflexion_agent([taskInfo, thinking4, thinking5], cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent.id}, verify intermediate results, thinking: {thinking6.content}; answer: {answer6.content}")
    critic_inst_6 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback6, correct6 = await critic_agent([taskInfo, thinking6], critic_inst_6, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback6.content}; correct: {correct6.content}")
        if correct6.content == "True":
            break
        thinking6, answer6 = await reflexion_agent([taskInfo, thinking6, feedback6], cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {reflexion_agent.id}, refining verification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Stage 2

    # Subtask 7: Using Law of Cosines and known R, r, cos A, sin B sin C, express AB · AC symbolically
    cot_sc_instruction_7 = (
        "Sub-task 7: Using the Law of Cosines and the known values of R and r, along with the exact cos A derived previously, "
        "express the product AB · AC in terms of R, r, cos A, and sin B sin C. Manipulate the expressions symbolically to isolate AB · AC without introducing approximations or assumptions about angle equality. "
        "Maintain all expressions in exact form."
    )
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", thinking5.content, thinking6.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_7 = []
    possible_thinkings_7 = []
    for i in range(self.max_sc):
        thinking7, answer7 = await cot_sc_agents[i]([taskInfo, thinking5, thinking6], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, express AB·AC symbolically, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7_final, answer7_final = await final_decision_agent_7([taskInfo] + possible_thinkings_7, "Sub-task 7: Synthesize and choose the most consistent symbolic expression for AB · AC.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize AB·AC symbolic expression, thinking: {thinking7_final.content}; answer: {answer7_final.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7_final.content}; answer - {answer7_final.content}")
    subtask_desc7['response'] = {"thinking": thinking7_final, "answer": answer7_final}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    # Subtask 8: Compute exact numerical value of AB · AC substituting R=13, r=6, derived cos A, sin B sin C
    reflect_inst_8 = (
        "Sub-task 8: Compute the exact numerical value of AB · AC by substituting R = 13, r = 6, and the derived expressions for cos A and sin B sin C. "
        "Avoid any rounding or decimal approximations; present the final answer as an exact fraction or simplified radical form. "
        "Verify that the result is consistent with all geometric constraints, especially the perpendicularity condition IA ⟂ OI."
    )
    cot_reflect_instruction_8 = "Sub-task 8: Your problem is to compute the exact numeric value." + " Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", thinking7_final.content],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await reflexion_agent([taskInfo, thinking7_final], cot_reflect_instruction_8, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent.id}, compute exact numeric AB·AC, thinking: {thinking8.content}; answer: {answer8.content}")
    critic_inst_8 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback8, correct8 = await critic_agent([taskInfo, thinking8], critic_inst_8, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback8.content}; correct: {correct8.content}")
        if correct8.content == "True":
            break
        thinking8, answer8 = await reflexion_agent([taskInfo, thinking8, feedback8], cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {reflexion_agent.id}, refining numeric computation, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
