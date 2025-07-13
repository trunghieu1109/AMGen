async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify and list all given geometric elements and their properties from the problem statement. "
        "Explicitly state the dimensions of rectangles ABCD and EFGH, the collinearity condition of points D, E, C, F, "
        "and the cyclic condition of points A, D, H, G. Emphasize the known side lengths and geometric constraints without performing any calculations or making assumptions about orientation or coordinate placement."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing given geometric elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Establish a consistent coordinate system for rectangle ABCD based on its properties and given side lengths. "
        "Define coordinates for points A, B, C, and D, ensuring the orientation respects the rectangle's properties and facilitates further analysis. "
        "Avoid arbitrary assumptions about the relative position of rectangle EFGH at this stage."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, establishing ABCD coordinates, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent coordinate system for ABCD." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Establish a consistent coordinate system for rectangle EFGH based on its properties and given side lengths. "
        "Define coordinates for points E, F, G, and H, ensuring the orientation respects the rectangle's properties and is compatible with the coordinate system of ABCD. "
        "Avoid assumptions about relative placement beyond what is necessary to represent EFGH."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, establishing EFGH coordinates, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent coordinate system for EFGH." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Formulate the equation of the line containing points D, E, C, and F using the collinearity condition. "
        "Express the coordinates of these points in terms of the established coordinate systems from subtasks 2 and 3. "
        "Set up parametric or explicit linear relationships without solving for unknowns yet. Avoid premature assumptions about the order of points on the line."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking2.content, thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, thinking3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, formulating collinearity line, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent line equation for points D, E, C, F." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Express the cyclic quadrilateral condition for points A, D, H, and G in terms of coordinate geometry. "
        "Set up the necessary equations or constraints using circle properties (e.g., concyclicity conditions such as equal opposite angles or the equation of the circle passing through these points). "
        "Do not solve the system at this stage; focus on correctly formulating the constraints."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking2.content, thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking2, thinking3], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, formulating cyclic condition, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent cyclic quadrilateral constraints." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_6 = (
        "Sub-task 6: Use the rectangle properties, coordinate placements, collinearity, and cyclic conditions to derive a system of equations involving unknown parameters. "
        "Solve this system symbolically or exactly, avoiding numeric approximations. Produce all candidate solutions explicitly, preserving exact radical expressions where necessary. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", thinking4.content, thinking5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4, thinking5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking4, thinking5] + all_thinking6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving system, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1], "Sub-task 6: Given all the above thinking and answers, reason over them carefully and provide a final set of candidate solutions." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_sc_instruction_6_5 = (
        "Sub-task 6.5: Verify and reconcile multiple candidate solutions obtained in Sub-task 6. "
        "Check geometric plausibility by confirming the order of points D, E, C, F on the line, ensuring all lengths are positive and consistent with rectangle and cyclic conditions. "
        "Select the unique solution that satisfies all geometric constraints. Document the reasoning and justification for the selection."
    )
    cot_agents_6_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_6_5 = []
    possible_thinkings_6_5 = []
    subtask_desc6_5 = {
        "subtask_id": "subtask_6_5",
        "instruction": cot_sc_instruction_6_5,
        "context": ["user query", thinking6.content, answer6.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6_5, answer6_5 = await cot_agents_6_5[i]([taskInfo, thinking6, answer6], cot_sc_instruction_6_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6_5[i].id}, verifying candidate solutions, thinking: {thinking6_5.content}; answer: {answer6_5.content}")
        possible_answers_6_5.append(answer6_5)
        possible_thinkings_6_5.append(thinking6_5)
    final_decision_agent_6_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6_5, answer6_5 = await final_decision_agent_6_5([taskInfo] + possible_thinkings_6_5, "Sub-task 6.5: Synthesize and select the unique valid solution among candidates." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 6.5 output: thinking - {thinking6_5.content}; answer - {answer6_5.content}")
    subtask_desc6_5['response'] = {"thinking": thinking6_5, "answer": answer6_5}
    logs.append(subtask_desc6_5)
    print("Step 6.5: ", sub_tasks[-1])

    debate_instruction_7 = (
        "Sub-task 7: Verify and reconcile multiple candidate solutions obtained in Sub-task 6.5. "
        "Check geometric plausibility by confirming the order of points D, E, C, F on the line, ensuring all lengths are positive and consistent with rectangle and cyclic conditions. "
        "Discard any solutions that violate these constraints. Select the unique solution that satisfies all geometric conditions. Document the reasoning and justification for the selection. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", thinking6_5.content, answer6_5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6_5, answer6_5], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6_5, answer6_5] + all_thinking7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying solutions, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1], "Sub-task 7: Given all the above thinking and answers, reason over them carefully and provide the unique valid solution." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_sc_instruction_8 = (
        "Sub-task 8: Calculate the length CE using the coordinates of points C and E from the verified solution in Sub-task 7. "
        "Use the exact symbolic expressions and the distance formula. Verify the calculation is consistent with all geometric constraints and the relative positions of points. "
        "Avoid any numeric approximations or rounding. Present the final length CE in exact radical form if it is not an integer."
    )
    cot_agents_8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_8 = []
    possible_thinkings_8 = []
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_sc_instruction_8,
        "context": ["user query", thinking7.content, answer7.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking8, answer8 = await cot_agents_8[i]([taskInfo, thinking7, answer7], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_8[i].id}, calculating CE length, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers_8.append(answer8)
        possible_thinkings_8.append(thinking8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + possible_thinkings_8, "Sub-task 8: Synthesize and provide the exact length CE." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    reflect_inst_9 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_9 = (
        "Sub-task 9: Perform a final reflexion and verification of the entire solution workflow. "
        "Confirm that no irrational quantities have been prematurely approximated or rounded. Check whether the radicand in any radical expressions is a perfect square. "
        "Ensure the final answer for CE is presented exactly and justified by the geometric constraints. If inconsistencies or approximations are found, recommend revisiting previous subtasks. "
        + reflect_inst_9
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round
    cot_inputs_9 = [taskInfo, thinking8, answer8]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction_9,
        "context": ["user query", thinking8.content, answer8.content],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, final verification, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max_9):
        feedback9, correct9 = await critic_agent_9([taskInfo, thinking9], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs_9.extend([thinking9, feedback9])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining verification, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
