async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Assign a coordinate system and place triangle ABC in the plane using the given side lengths AB=5, BC=9, and AC=10. "
        "Find coordinates of points A, B, and C accordingly, ensuring the triangle is valid and consistent with the given lengths."
    )
    subtask_id_1 = "subtask_1"
    print(f"Logging before agent call: subtask_id={subtask_id_1}, instruction={cot_instruction_1}, context=[user query], agent_collaboration=CoT")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, assigning coordinates and finding points A,B,C, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1 = {
        "subtask_id": subtask_id_1,
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT",
        "response": {"thinking": thinking1, "answer": answer1}
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on coordinates of A, B, and C, determine the equation of the circumcircle ω passing through these points. "
        "Use self-consistency by having multiple agents independently verify the circumcircle equation and confirm it passes through A, B, and C."
    )
    subtask_id_2 = "subtask_2"
    print(f"Logging before agent call: subtask_id={subtask_id_2}, instruction={cot_sc_instruction_2}, context=[user query, thinking of subtask 1, answer of subtask 1], agent_collaboration=SC_CoT")
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining circumcircle, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2 = {
        "subtask_id": subtask_id_2,
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT",
        "response": {"thinking": thinking2, "answer": answer2}
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Find the equations of the tangents to the circumcircle ω at points B and C, and calculate the coordinates of point D, the intersection of these tangents. "
        "Use self-consistency by multiple agents to verify tangent correctness and intersection point D."
    )
    subtask_id_3 = "subtask_3"
    print(f"Logging before agent call: subtask_id={subtask_id_3}, instruction={cot_sc_instruction_3}, context=[user query, thinking of subtask 2, answer of subtask 2], agent_collaboration=SC_CoT")
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, finding tangents and point D, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3 = {
        "subtask_id": subtask_id_3,
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT",
        "response": {"thinking": thinking3, "answer": answer3}
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_reflect_instruction_4 = (
        "Sub-task 4: Using synthetic geometry, recall and state the key properties of point D as the intersection of tangents at B and C, including the pole-polar relationship and harmonic division involving points A, P, B, C, and D. "
        "Apply the power of a point theorem at D to relate segments DA, DP, DB, and DC, and derive an expression for AP. "
        "Use reflexion to validate these properties and ensure consistency with coordinate geometry results."
    )
    subtask_id_4 = "subtask_4"
    print(f"Logging before agent call: subtask_id={subtask_id_4}, instruction={cot_reflect_instruction_4}, context=[user query, thinking of subtask 3, answer of subtask 3], agent_collaboration=Reflexion")
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, recalling pole-polar and harmonic division properties, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                               "Please review the synthetic geometry properties and harmonic division reasoning for point D and P, and provide limitations or corrections.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining synthetic geometry reasoning, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4 = {
        "subtask_id": subtask_id_4,
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion",
        "response": {"thinking": thinking4, "answer": answer4}
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Using coordinates of A and D, find the equation of line AD. "
        "Then find the second intersection point P of line AD with the circumcircle ω (other than A) by solving the system of equations. "
        "Use self-consistency with multiple agents to verify the correctness of point P."
    )
    subtask_id_5 = "subtask_5"
    print(f"Logging before agent call: subtask_id={subtask_id_5}, instruction={cot_sc_instruction_5}, context=[user query, thinking of subtask 3, answer of subtask 3, thinking of subtask 2, answer of subtask 2, thinking of subtask 1, answer of subtask 1], agent_collaboration=SC_CoT")
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking3, answer3, thinking2, answer2, thinking1, answer1], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, finding line AD and point P, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    answer5_content = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[answer5_content]
    answer5 = answermapping_5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5 = {
        "subtask_id": subtask_id_5,
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT",
        "response": {"thinking": thinking5, "answer": answer5}
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction_6 = (
        "Sub-task 6: Cross-validate the synthetic geometry result for AP (from subtask 4) with the coordinate geometry result (from subtask 5) by comparing numeric approximations. "
        "If discrepancies arise, revisit previous steps to identify and correct errors; otherwise, confirm consistency and proceed."
    )
    subtask_id_6 = "subtask_6"
    print(f"Logging before agent call: subtask_id={subtask_id_6}, instruction={cot_reflect_instruction_6}, context=[user query, thinking of subtask 4, answer of subtask 4, thinking of subtask 5, answer of subtask 5], agent_collaboration=Reflexion")
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking4, answer4, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, cross-validating synthetic and coordinate results, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6],
                                               "Please review the cross-validation of AP length from synthetic and coordinate geometry and provide any inconsistencies or confirm correctness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining cross-validation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {
        "subtask_id": subtask_id_6,
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion",
        "response": {"thinking": thinking6, "answer": answer6}
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    debate_instruction_7 = (
        "Sub-task 7: Calculate the length of segment AP using the coordinates of A and P. "
        "Express length AP as a simplified fraction m/n where m and n are relatively prime integers, and compute m + n as the final answer. "
        "Use debate among agents to ensure correctness, simplification, and plausibility of the final numeric answer."
    )
    subtask_id_7 = "subtask_7"
    print(f"Logging before agent call: subtask_id={subtask_id_7}, instruction={debate_instruction_7}, context=[user query, thinking of subtask 5, answer of subtask 5], agent_collaboration=Debate")
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking5, answer5], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking5, answer5] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating AP length and simplifying fraction, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on simplified fraction and sum m+n.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final answer m+n, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7 = {
        "subtask_id": subtask_id_7,
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate",
        "response": {"thinking": thinking7, "answer": answer7}
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
