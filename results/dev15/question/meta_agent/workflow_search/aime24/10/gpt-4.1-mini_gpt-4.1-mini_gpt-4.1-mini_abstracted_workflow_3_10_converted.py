async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Assign a coordinate system and define coordinates for points A, B, C, and D of rectangle ABCD "
        "using the given side lengths AB=107 and BC=16. Ensure rectangle properties (right angles, opposite sides equal) are satisfied. "
        "Avoid assuming any orientation or position beyond what is necessary, and do not fix the rectangle's placement relative to the other rectangle at this stage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, assigning coordinates for ABCD, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Assign coordinates for points E, F, G, and H of rectangle EFGH using the given side lengths EF=184 and FG=17. "
        "Ensure rectangle properties hold. Avoid assuming any relative position or orientation with respect to rectangle ABCD or any other points at this stage."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, assigning coordinates for EFGH, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Express the collinearity condition of points D, E, C, and F as a general linear equation relating their coordinates. "
        "Parameterize the line containing these points without assuming any specific orientation (e.g., not necessarily horizontal or vertical). "
        "Clearly state the parametric or implicit form of the line and express coordinates of E and F in terms of this parameterization. "
        "Avoid premature assumptions about point order or line slope."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content, thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, thinking2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, formulating collinearity condition, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Derive the concyclicity condition for points A, D, H, and G by rigorously formulating the equation of the circle passing through these points. "
        "Use the standard determinant condition for concyclicity or solve the circle equation (center (h,k) and radius r) with all four points. "
        "Do not assume any special properties or locations for the circle center (e.g., do not fix it on x=0). "
        "Incorporate the coordinates and line equation from Sub-task 3 to cross-check geometric consistency."
    )
    N_sc = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking1.content, thinking2.content, thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1, thinking2, thinking3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, deriving concyclicity condition, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent concyclicity condition." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Combine the rectangle properties, the general collinearity parameterization from Sub-task 3, "
        "and the concyclicity condition from Sub-task 4 to derive a complete system of equations relating the coordinates of points C and E (and other relevant points). "
        "Avoid any premature simplifications or assumptions about relative positions or orientations. Present the system clearly, specifying all variables and constraints."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking3.content, thinking4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking3, thinking4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, combining constraints, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent system of equations." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6a = (
        "Sub-task 6a: Using the collinearity parameterization from Sub-task 3, express the coordinates of points E and F explicitly in terms of a parameter (e.g., t), "
        "ensuring the parameterization is consistent with the rectangle properties and given side lengths. Prepare these expressions to be used in solving the full system. "
        "Do not assume the collinearity line is horizontal or that E and F share the same y-coordinate as D and C."
    )
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking3], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, parameterizing E and F, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])

    reflect_inst_6b = (
        "Sub-task 6b: Solve the full system of equations obtained in Sub-task 5, incorporating the parameterized coordinates of E and F from Sub-task 6a, "
        "to find explicit coordinates or relative positions of points C and E. Ensure that all solutions satisfy the original geometric constraints, including rectangle properties, collinearity, and concyclicity, without any unjustified assumptions. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_6b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6b = self.max_round
    cot_inputs_6b = [taskInfo, thinking5, thinking6a]
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": reflect_inst_6b,
        "context": ["user query", thinking5.content, thinking6a.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6b, answer6b = await cot_agent_6b(cot_inputs_6b, reflect_inst_6b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6b.id}, solving system, thinking: {thinking6b.content}; answer: {answer6b.content}")
    for i in range(N_max_6b):
        feedback, correct = await critic_agent_6b([taskInfo, thinking6b], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6b.extend([thinking6b, feedback])
        thinking6b, answer6b = await cot_agent_6b(cot_inputs_6b, reflect_inst_6b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6b.id}, refining solution, thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])

    cot_sc_instruction_7 = (
        "Sub-task 7: Calculate the length of segment CE using the coordinates or relative positions found in Sub-task 6b. "
        "Verify that the computed length is consistent with all given conditions and constraints, including the collinearity of D, E, C, F and the concyclicity of A, D, H, G. "
        "Confirm that no assumptions about line orientation or circle center location contradict the solution."
    )
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_7 = []
    possible_thinkings_7 = []
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", thinking6b.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking6b], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, calculating length CE, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + possible_thinkings_7, "Sub-task 7: Synthesize and choose the most consistent length CE." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
