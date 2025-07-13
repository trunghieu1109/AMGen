async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1a = (
        "Sub-task 1a: Establish a base coordinate system by placing vertices A, B, and C in 3D space consistent with the given edge lengths: "
        "AB = sqrt(41), AC = sqrt(80), BC = sqrt(89). Assign explicit coordinates to A, B, and C to fix the reference frame. "
        "Verify that these placements satisfy the given distances exactly and that triangle ABC is non-degenerate. "
        "Explain your reasoning step-by-step."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, placing A, B, C coordinates, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_instruction_1b = (
        "Sub-task 1b: Determine the coordinates of vertex D by solving the system of equations derived from the given edge lengths: "
        "AD = sqrt(89), BD = sqrt(80), CD = sqrt(41), using the fixed coordinates of A, B, and C from Sub-task 1a. "
        "Provide exact or numeric solutions for D's coordinates. Verify that the solution is consistent and that the tetrahedron ABCD is non-degenerate. "
        "If coordinate determination is overly complex, prepare to use Cayley–Menger determinant directly in later subtasks. "
        "Explain your reasoning step-by-step."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", thinking1a.content, answer1a.content],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, solving for D coordinates, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Using the vertex coordinates from Sub-tasks 1a and 1b or directly from edge lengths, compute the areas of the four faces ABC, ABD, ACD, and BCD. "
        "Use Heron's formula or vector cross products. Maintain exact symbolic expressions (e.g., radicals) rather than decimal approximations. "
        "Verify that all areas are positive and consistent with the tetrahedron's geometry. Provide detailed calculations and reasoning."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1a.content, answer1a.content, thinking1b.content, answer1b.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, computing face areas, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Calculate the volume of tetrahedron ABCD using the Cayley–Menger determinant formula based solely on the six given edge lengths: "
        "AB = sqrt(41), AC = sqrt(80), AD = sqrt(89), BC = sqrt(89), BD = sqrt(80), CD = sqrt(41). "
        "Avoid reliance on vertex coordinates to prevent propagation of coordinate errors. Provide an exact or simplified radical expression for the volume. "
        "Perform a numerical sanity check to ensure the volume is positive and reasonable relative to the face areas. Provide detailed reasoning and calculations."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1a.content, answer1a.content, thinking1b.content, answer1b.content, thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculating volume, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    best_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[best_answer_3]
    answer3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    reflect_instruction_4a = (
        "Sub-task 4a: Verify whether the tetrahedron ABCD is tangential, i.e., admits an inscribed sphere. "
        "Use necessary and sufficient conditions such as the existence of a point equidistant from all faces or relationships among face areas and edge lengths. "
        "Check if the inradius formula r = 3 * volume / (sum of face areas) is applicable. "
        "If tangentiality fails, report and halt further inradius computations. Provide detailed reasoning and verification."
    )
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4a = self.max_round
    cot_inputs_4a = [taskInfo, thinking2, answer2, thinking3, answer3]
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": reflect_instruction_4a,
        "context": ["user query", thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, reflect_instruction_4a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, verifying tangentiality, thinking: {thinking4a.content}; answer: {answer4a.content}")
    for i in range(N_max_4a):
        feedback4a, correct4a = await critic_agent_4a([taskInfo, thinking4a, answer4a],
                                                    "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.",
                                                    i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4a.id}, providing feedback, thinking: {feedback4a.content}; answer: {correct4a.content}")
        if correct4a.content == "True":
            break
        cot_inputs_4a.extend([thinking4a, answer4a, feedback4a])
        thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, reflect_instruction_4a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, refining tangentiality verification, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    if "no" in answer4a.content.lower() or "not" in answer4a.content.lower():
        final_answer = await self.make_final_answer(thinking4a, answer4a, sub_tasks, agents)
        return final_answer, logs

    cot_sc_instruction_4b = (
        "Sub-task 4b: Find the coordinates of the incenter point I inside tetrahedron ABCD such that the perpendicular distances from I to each of the four faces are equal. "
        "Formulate and solve the system of equations representing equal distances to the planes of the faces using vertex coordinates from Sub-task 1b. "
        "Confirm the uniqueness and existence of I inside the tetrahedron. Provide detailed reasoning and calculations."
    )
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query", thinking1b.content, answer1b.content, thinking4a.content, answer4a.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking1b, answer1b, thinking4a, answer4a], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, finding incenter coordinates, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    best_answer_4b = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[best_answer_4b]
    answer4b = answermapping_4b[best_answer_4b]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    cot_sc_instruction_4c = (
        "Sub-task 4c: Compute the perpendicular distances from the incenter I to each face using the coordinates of I and the equations of the faces. "
        "Verify that these distances are equal within exact or high-precision tolerance. This confirms the correctness of the incenter and justifies the use of the inradius formula. "
        "Provide detailed calculations and verification."
    )
    cot_agents_4c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4c = []
    thinkingmapping_4c = {}
    answermapping_4c = {}
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_sc_instruction_4c,
        "context": ["user query", thinking4b.content, answer4b.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4c, answer4c = await cot_agents_4c[i]([taskInfo, thinking4b, answer4b], cot_sc_instruction_4c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4c[i].id}, verifying equal distances, thinking: {thinking4c.content}; answer: {answer4c.content}")
        possible_answers_4c.append(answer4c.content)
        thinkingmapping_4c[answer4c.content] = thinking4c
        answermapping_4c[answer4c.content] = answer4c
    best_answer_4c = Counter(possible_answers_4c).most_common(1)[0][0]
    thinking4c = thinkingmapping_4c[best_answer_4c]
    answer4c = answermapping_4c[best_answer_4c]
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])

    cot_sc_instruction_4d = (
        "Sub-task 4d: Calculate the inradius r of the tetrahedron using the formula r = 3 * volume / (sum of face areas), "
        "only if tangentiality is confirmed and the incenter distances are equal. Express r symbolically and prepare for simplification. "
        "Provide detailed reasoning and calculations."
    )
    cot_agents_4d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4d = []
    thinkingmapping_4d = {}
    answermapping_4d = {}
    subtask_desc4d = {
        "subtask_id": "subtask_4d",
        "instruction": cot_sc_instruction_4d,
        "context": ["user query", thinking2.content, answer2.content, thinking3.content, answer3.content, thinking4a.content, answer4a.content, thinking4c.content, answer4c.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4d, answer4d = await cot_agents_4d[i]([taskInfo, thinking2, answer2, thinking3, answer3, thinking4a, answer4a, thinking4c, answer4c], cot_sc_instruction_4d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4d[i].id}, calculating inradius, thinking: {thinking4d.content}; answer: {answer4d.content}")
        possible_answers_4d.append(answer4d.content)
        thinkingmapping_4d[answer4d.content] = thinking4d
        answermapping_4d[answer4d.content] = answer4d
    best_answer_4d = Counter(possible_answers_4d).most_common(1)[0][0]
    thinking4d = thinkingmapping_4d[best_answer_4d]
    answer4d = answermapping_4d[best_answer_4d]
    sub_tasks.append(f"Sub-task 4d output: thinking - {thinking4d.content}; answer - {answer4d.content}")
    subtask_desc4d['response'] = {"thinking": thinking4d, "answer": answer4d}
    logs.append(subtask_desc4d)
    print("Step 4d: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Simplify the inradius expression r obtained in Sub-task 4d into the form (m*sqrt(n))/p, where m, n, p are positive integers, "
        "m and p are coprime, and n is square-free. Perform prime factorization and fraction reduction carefully to meet all problem constraints. "
        "Maintain exact symbolic manipulation to avoid rounding errors. Provide detailed reasoning and final simplified expression."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4d.content, answer4d.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4d, answer4d], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4d, answer4d] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying inradius, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Simplify inradius expression and provide final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Compute the sum m + n + p from the simplified inradius expression obtained in Sub-task 5. "
        "Verify correctness and provide the final numeric answer along with a summary of verification steps confirming the solution's validity and consistency with the problem statement."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, computing final sum, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
