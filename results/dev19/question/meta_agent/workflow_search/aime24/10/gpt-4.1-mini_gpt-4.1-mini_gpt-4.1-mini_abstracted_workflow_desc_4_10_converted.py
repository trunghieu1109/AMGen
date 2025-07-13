async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and formalize all given data and geometric properties from the problem statement. "
        "List all known side lengths of rectangles ABCD and EFGH, state rectangle properties (right angles, equal opposite sides), "
        "and note the collinearity of points D, E, C, F and concyclicity of points A, D, H, G. Avoid assumptions about relative positions or orientations."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    N_sc = self.max_sc
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    for i in range(N_sc):
        thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_0_1.id}, iteration {i}, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
        possible_answers_0_1.append(answer_0_1)
        possible_thinkings_0_1.append(thinking_0_1)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_thinkings_0_1,
        "Sub-task 1: Synthesize and choose the most consistent formalization of given data and properties.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Assign a coordinate system and represent rectangles ABCD and EFGH in this system consistent with their properties and given side lengths. "
        "Place rectangle ABCD first with explicit coordinates for points A, B, C, D based on given side lengths, ensuring right angles and side lengths are respected. "
        "Then represent rectangle EFGH with unknown horizontal coordinate x0 for point E, keeping vertical coordinates consistent with rectangle properties. "
        "Do not yet impose collinearity or concyclicity constraints."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": [taskInfo, thinking_0_1],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    for i in range(N_sc):
        thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_0_2.id}, iteration {i}, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2)
        possible_thinkings_0_2.append(thinking_0_2)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo] + possible_thinkings_0_2,
        "Sub-task 2: Synthesize and choose the most consistent coordinate assignment for rectangles.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_0_3 = (
        "Sub-task 3: Formulate the collinearity condition of points D, E, C, F as an explicit algebraic constraint in the chosen coordinate system. "
        "Express this condition as an equation relating the coordinates of these points, ensuring the order of points on the line is clearly stated or parameterized. "
        "Avoid assuming the order without justification."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": [taskInfo, thinking_0_2],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    for i in range(N_sc):
        thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_2], cot_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_0_3.id}, iteration {i}, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
        possible_answers_0_3.append(answer_0_3)
        possible_thinkings_0_3.append(thinking_0_3)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3(
        [taskInfo] + possible_thinkings_0_3,
        "Sub-task 3: Synthesize and choose the most consistent collinearity formulation.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_0_4 = (
        "Sub-task 4: Formulate the concyclicity condition of points A, D, H, G as an explicit algebraic constraint in the chosen coordinate system. "
        "Use the determinant or power-of-a-point condition to express that these four points lie on the same circle. Do not attempt to solve this condition yet; only formulate it clearly."
    )
    cot_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": cot_instruction_0_4,
        "context": [taskInfo, thinking_0_2],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_0_4 = []
    possible_thinkings_0_4 = []
    for i in range(N_sc):
        thinking_0_4, answer_0_4 = await cot_agent_0_4([taskInfo, thinking_0_2], cot_instruction_0_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_0_4.id}, iteration {i}, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
        possible_answers_0_4.append(answer_0_4)
        possible_thinkings_0_4.append(thinking_0_4)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4(
        [taskInfo] + possible_thinkings_0_4,
        "Sub-task 4: Synthesize and choose the most consistent concyclicity formulation.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_1_1 = (
        "Sub-task 1: Analyze and verify the relative positions and orientations of rectangles ABCD and EFGH that satisfy the collinearity condition of D, E, C, F and the concyclicity condition of A, D, H, G. "
        "Confirm compatibility of these conditions with the given side lengths and coordinate assignments. Avoid premature numeric assumptions; focus on logical consistency and feasibility. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": [taskInfo, thinking_0_3, thinking_0_4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking_1_1, answer_1_1 = await agent([taskInfo, thinking_0_3, thinking_0_4], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking_0_3, thinking_0_4] + all_thinking_1_1[r-1]
                thinking_1_1, answer_1_1 = await agent(input_infos_1_1, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
            all_thinking_1_1[r].append(thinking_1_1)
            all_answer_1_1[r].append(answer_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0_3, thinking_0_4] + all_thinking_1_1[-1],
        "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_1_2 = (
        "Sub-task 2: Enumerate and analyze possible orders of points D, E, C, F on the line, considering the geometric constraints and rectangle properties. "
        "Determine which order(s) are consistent with the collinearity and concyclicity conditions and the given side lengths. Avoid assuming a single order without justification. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": [taskInfo, thinking_0_3, thinking_1_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_1_2, answer_1_2 = await agent([taskInfo, thinking_0_3, thinking_1_1], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos_1_2 = [taskInfo, thinking_0_3, thinking_1_1] + all_thinking_1_2[r-1]
                thinking_1_2, answer_1_2 = await agent(input_infos_1_2, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
            all_thinking_1_2[r].append(thinking_1_2)
            all_answer_1_2[r].append(answer_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking_0_3, thinking_1_1] + all_thinking_1_2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Derive the explicit algebraic equation(s) from the concyclicity condition formulated earlier to solve for the unknown horizontal coordinate x0 of point E, given its fixed vertical coordinate. "
        "Perform symbolic or numeric solving of this equation, ensuring all geometric constraints are respected. Avoid heuristic or shortcut numeric substitutions; rigorously solve the equation."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": [taskInfo, thinking_0_4, thinking_1_2],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_0_4, thinking_1_2], cot_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_2_1.id}, iteration {i}, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo, thinking_0_4, thinking_1_2] + possible_thinkings_2_1,
        "Sub-task 1: Synthesize and choose the most consistent solution for x0 from the concyclicity condition.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 7: ", sub_tasks[-1])

    reflect_instruction_2_2 = (
        "Sub-task 2: Using the solved value(s) of x0 from the previous subtask, compute the exact length of segment CE as the absolute difference of the horizontal coordinates of points C and E. "
        "Ensure this computation strictly uses the solved coordinate without any heuristic simplifications or assumptions. Validate the numeric result for consistency with all constraints. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflect_instruction_2_2,
        "context": [taskInfo, thinking_2_1],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, initial thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback_2_2.content}; correct: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refined thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 8: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1: Review and validate the entire reasoning chain and numeric results, focusing on the correctness of the concyclicity solving step and the final CE length computation. "
        "Engage in reflexion or debate to identify any overlooked assumptions or errors. Confirm that the final answer respects all problem conditions and geometric constraints before finalizing. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": [taskInfo, thinking_2_1, thinking_2_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_3_1, answer_3_1 = await agent([taskInfo, thinking_2_1, thinking_2_2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking_2_1, thinking_2_2] + all_thinking_3_1[r-1]
                thinking_3_1, answer_3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
            all_thinking_3_1[r].append(thinking_3_1)
            all_answer_3_1[r].append(answer_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1(
        [taskInfo, thinking_2_1, thinking_2_2] + all_thinking_3_1[-1],
        "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
