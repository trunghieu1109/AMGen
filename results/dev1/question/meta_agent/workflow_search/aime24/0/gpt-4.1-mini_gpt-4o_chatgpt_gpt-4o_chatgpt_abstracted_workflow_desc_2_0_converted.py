async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Assign coordinates to points A, B, and C of triangle ABC with AB=5, BC=9, AC=10. "
        "Place B at (0,0), C at (9,0), and find A coordinates accordingly. "
        "Explain the coordinate choices and calculations. Provide coordinates as a dictionary { 'A': [x,y], 'B': [x,y], 'C': [x,y] } in JSON format."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_1: {subtask_desc1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, assigning coordinates of ABC, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Using coordinates from Sub-task 1, formulate the general equation of the circumcircle omega in the form x^2 + y^2 + D x + E y + F = 0. "
        "Set up a linear system by substituting coordinates of A, B, and C to solve for D, E, and F. "
        "Provide the solved values of D, E, and F as a JSON dictionary { 'D': value, 'E': value, 'F': value } with exact or simplified fractions if possible."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_2: {subtask_desc2}")
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, solving circle parameters D,E,F, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Compute the center coordinates (-D/2, -E/2) and radius of the circumcircle omega from the solved parameters D, E, and F. "
        "Provide the center as a dictionary { 'center': [x, y] } and radius as a number."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_3: {subtask_desc3}")
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, computing center and radius, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Verify the correctness of the circumcircle by checking that the distances from the computed center to points A, B, and C are equal within a small tolerance (e.g., 1e-6). "
        "If verification fails, indicate failure and suggest revising the circle computation. "
        "Provide verification result as { 'verified': true/false, 'distances': [dA, dB, dC] } with distances as numbers."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3, answer3, thinking1, answer1],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_4: {subtask_desc4}")
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, verifying circumcircle correctness, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Find the equations of the tangent lines to the circumcircle omega at points B and C. "
        "Use the fact that the tangent at a point on the circle is perpendicular to the radius drawn to that point. "
        "Provide tangent line equations in slope-intercept form or parametric form as JSON, e.g., { 'tangent_B': { 'slope': m1, 'intercept': b1 }, 'tangent_C': { 'slope': m2, 'intercept': b2 } } or parametric equivalents."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking3, answer3, thinking1, answer1],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_5: {subtask_desc5}")
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3, thinking1, answer1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, finding tangent lines at B and C, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Verify that each tangent line is perpendicular to the radius at its point of tangency by checking the dot product of their direction vectors is zero within tolerance. "
        "Provide verification result as { 'verified_B': true/false, 'verified_C': true/false } with explanations."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5, answer5, thinking3, answer3],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_6: {subtask_desc6}")
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5, thinking3, answer3], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, verifying tangent perpendicularity, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Determine the coordinates of point D, the intersection of the tangent lines at B and C, by solving the system of tangent line equations. "
        "Provide coordinates as { 'D': [x, y] } in JSON format."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking5, answer5],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_7: {subtask_desc7}")
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking5, answer5], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, finding intersection D of tangents, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Find the parametric or explicit equation of line AD using coordinates of points A and D. "
        "Provide the line equation or parametric form as JSON, e.g., { 'line_AD': { 'point': [x0, y0], 'direction': [dx, dy] } } or slope-intercept form."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking7, answer7, thinking1, answer1],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_8: {subtask_desc8}")
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7, thinking1, answer1], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, finding line AD equation, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction_9 = (
        "Sub-task 9: Find the second intersection point P (other than A) of line AD with the circumcircle omega by solving the system of the line and circle equations. "
        "Provide coordinates of P as { 'P': [x, y] } in JSON format."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", thinking8, answer8, thinking2, answer2],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_9: {subtask_desc9}")
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking8, answer8, thinking2, answer2], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, finding point P on AD and omega, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    cot_instruction_10 = (
        "Sub-task 10: Calculate the length AP using the coordinates of points A and P obtained from previous subtasks. "
        "Show the distance formula application and simplification. Provide length AP as a number or simplified radical form."
    )
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction_10,
        "context": ["user query", thinking9, answer9, thinking1, answer1],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_10: {subtask_desc10}")
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking9, answer9, thinking1, answer1], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, calculating length AP, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    cot_instruction_11 = (
        "Sub-task 11: Check whether the length AP is a rational number or can be expressed as a fraction of relatively prime integers. "
        "If irrational, flag for review or alternative reasoning. Provide result as { 'is_rational': true/false, 'reason': explanation }."
    )
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_instruction_11,
        "context": ["user query", thinking10, answer10],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_11: {subtask_desc11}")
    thinking11, answer11 = await cot_agent_11([taskInfo, thinking10, answer10], cot_instruction_11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_11.id}, checking rationality of AP, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])

    cot_instruction_12 = (
        "Sub-task 12: Express the length AP in simplest fractional form m/n where m and n are relatively prime integers, and compute the sum m + n. "
        "If AP is irrational, explain why and suggest revisiting earlier steps. Provide final fraction and sum as { 'm': m, 'n': n, 'sum': m_plus_n } or explanation."
    )
    cot_agent_12 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc12 = {
        "subtask_id": "subtask_12",
        "instruction": cot_instruction_12,
        "context": ["user query", thinking11, answer11, thinking10, answer10],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_12: {subtask_desc12}")
    thinking12, answer12 = await cot_agent_12([taskInfo, thinking11, answer11, thinking10, answer10], cot_instruction_12, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_12.id}, expressing AP as fraction and summing m+n, thinking: {thinking12.content}; answer: {answer12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    subtask_desc12['response'] = {"thinking": thinking12, "answer": answer12}
    logs.append(subtask_desc12)
    print("Step 12: ", sub_tasks[-1])

    cot_reflect_instruction_13 = (
        "Sub-task 13: Reflect on and verify the final answer for consistency with the problem’s conditions, including the form of AP and the reasonableness of the computed value. "
        "If inconsistencies arise, suggest revisiting earlier subtasks. Provide final verification as { 'consistent': true/false, 'comments': explanation }."
    )
    cot_agent_13 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_13 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_13 = self.max_round
    cot_inputs_13 = [taskInfo, thinking12, answer12]
    subtask_desc13 = {
        "subtask_id": "subtask_13",
        "instruction": cot_reflect_instruction_13,
        "context": ["user query", thinking12, answer12],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before subtask_13: {subtask_desc13}")
    thinking13, answer13 = await cot_agent_13(cot_inputs_13, cot_reflect_instruction_13, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_13.id}, verifying final answer consistency, thinking: {thinking13.content}; answer: {answer13.content}")
    for i in range(N_max_13):
        feedback, correct = await critic_agent_13([taskInfo, thinking13, answer13], "Please review the final answer consistency and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_13.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_13.extend([thinking13, answer13, feedback])
        thinking13, answer13 = await cot_agent_13(cot_inputs_13, cot_reflect_instruction_13, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_13.id}, refining final answer consistency, thinking: {thinking13.content}; answer: {answer13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking13.content}; answer - {answer13.content}")
    subtask_desc13['response'] = {"thinking": thinking13, "answer": answer13}
    logs.append(subtask_desc13)
    print("Step 13: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking13, answer13, sub_tasks, agents)
    return final_answer, logs
