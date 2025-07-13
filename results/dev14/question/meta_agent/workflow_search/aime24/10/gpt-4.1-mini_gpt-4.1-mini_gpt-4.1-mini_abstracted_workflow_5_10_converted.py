async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Extract and represent the given side lengths and geometric properties of rectangles ABCD and EFGH. "
        "Specifically, record that ABCD and EFGH are rectangles with right angles and equal opposite sides, and note the side lengths AB=107, BC=16, EF=184, and FG=17. "
        "Avoid making any assumptions about the orientation or coordinate placement of the rectangles at this stage."
    )
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
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, extract rectangle properties, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1, "Sub-task 1: Synthesize and choose the most consistent answer for rectangle properties.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Formulate the collinearity condition of points D, E, C, and F as a general linear geometric constraint. "
        "Express their coordinates or parametric relations along a single line with unknown slope and intercept parameters. "
        "Do not assign specific numeric coordinates or fix the line orientation; keep the line parametrization fully general to avoid premature assumptions."
    )
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, formulate collinearity parametrization, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent collinearity parametrization.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Express the cyclic quadrilateral condition for points A, D, H, and G. "
        "Identify the algebraic form of the concyclicity condition (e.g., the circle equation passing through these points or the determinant condition for concyclicity) without solving for coordinates yet. "
        "Emphasize the geometric implications and the equations that must be satisfied by their coordinates."
    )
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_sc_agents_3[i]([taskInfo, thinking1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, formulate cyclic quadrilateral condition, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent cyclic quadrilateral condition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Choose a suitable coordinate system to place rectangle ABCD. "
        "Place point A at the origin and side AB along the positive x-axis to fix a reference frame. "
        "Do not fix the orientation or position of rectangle EFGH or the collinearity line at this stage. "
        "Clearly state that the collinearity line D-E-C-F remains parametrized with unknown slope and parameters, preserving generality."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking1.content, thinking2.content, thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, thinking2, thinking3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, choose coordinate system for ABCD, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Express the coordinates of points B, C, and D of rectangle ABCD based on the chosen coordinate system and known side lengths. "
        "Similarly, express points E, F, G, and H of rectangle EFGH symbolically, using unknown parameters for their positions and orientations. "
        "Maintain the parametric form of the collinearity line for points D, E, C, and F, and represent these points accordingly without fixing numeric values."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, express coordinates of points, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Apply the collinearity condition to points D, E, C, and F using their parametric coordinates. "
        "Derive explicit linear equations relating the unknown parameters that define the positions of E and F on the collinearity line. "
        "Avoid fixing the line orientation; keep slope and intercept as variables. This subtask isolates the collinearity constraint algebraically."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking2.content, thinking5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking2, thinking5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, apply collinearity condition algebraically, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Formulate the cyclic quadrilateral condition algebraically for points A, D, H, and G using their coordinates. "
        "Write out the circle equation or the determinant condition for concyclicity explicitly in terms of unknown parameters. "
        "This subtask prepares the system of equations needed to enforce the cyclic condition without numeric assumptions."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking3.content, thinking5.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking3, thinking5], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, formulate cyclic quadrilateral algebraic condition, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Combine the equations derived from the collinearity condition (subtask_6), the cyclic quadrilateral condition (subtask_7), and the rectangle side length constraints to form a complete system of equations. "
        "Solve this system symbolically or numerically for all unknown parameters, including the slope and intercept of the collinearity line and the coordinates of points E, F, G, and H. "
        "Avoid assumptions or guesses; carry forward_10 explicit numeric or symbolic solutions."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking6.content, thinking7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking6, thinking7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, solve system of equations, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    reflect_inst_9 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_9 = "Sub-task 9: Verify the solutions obtained in subtask_8 by checking the concyclicity condition numerically (e.g., by computing the residual of the circle equation or determinant) and confirming the collinearity of points D, E, C, and F within a negligible error margin. " + reflect_inst_9
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round
    cot_inputs_9 = [taskInfo, thinking8]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction_9,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, verify and refine solution, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max_9):
        feedback, correct = await critic_agent_9([taskInfo, thinking9], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_9.extend([thinking9, feedback])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining solution, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    reflect_inst_10 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_10 = "Sub-task 10: Using the verified coordinates of points C and E from subtask_8, compute the length CE by applying the distance formula. " + reflect_inst_10
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_10 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_10 = self.max_round
    cot_inputs_10 = [taskInfo, thinking9]
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_reflect_instruction_10,
        "context": ["user query", thinking9.content],
        "agent_collaboration": "Reflexion"
    }
    thinking10, answer10 = await cot_agent_10(cot_inputs_10, cot_reflect_instruction_10, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_10.id}, compute length CE, thinking: {thinking10.content}; answer: {answer10.content}")
    for i in range(N_max_10):
        feedback, correct = await critic_agent_10([taskInfo, thinking10], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_10.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_10.extend([thinking10, feedback])
        thinking10, answer10 = await cot_agent_10(cot_inputs_10, cot_reflect_instruction_10, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_10.id}, refining length CE calculation, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
