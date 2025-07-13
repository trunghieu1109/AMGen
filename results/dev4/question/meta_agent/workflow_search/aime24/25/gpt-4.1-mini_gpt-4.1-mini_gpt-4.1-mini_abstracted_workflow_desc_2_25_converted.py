async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Derive a precise vector or coordinate representation of the convex equilateral hexagon ABCDEF with the property that all pairs of opposite sides are parallel. "
        "Assign coordinates or vectors to vertices A, B, C, D, E, and F with unknown parameters representing side length and angles. "
        "Ensure all six sides have equal length and that opposite sides are parallel (AB || DE, BC || EF, CD || FA). "
        "Carefully consider convexity constraints and avoid assuming the hexagon is regular (equal angles). "
        "Do not assign numeric values to side length or angles at this stage. Document all vector relations and constraints explicitly to support subsequent derivations."
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
    agents.append(f"CoT agent {cot_agent_1.id}, deriving vector representation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2a = (
        "Sub-task 2a: Formally define parametric equations for the lines containing sides AB, CD, and EF using the vector or coordinate representation derived in subtask_1. "
        "Express each line in parametric form with parameters representing points along the line. Avoid assuming any special properties beyond those established in subtask_1. "
        "This step sets the foundation for precise intersection calculations."
    )
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_2a: {subtask_desc2a}")
    thinking2a, answer2a = await cot_agent_2a([taskInfo, thinking1, answer1], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, defining parametric line equations, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])

    cot_instruction_2b = (
        "Sub-task 2b: Compute the intersection points P = AB ∩ CD, Q = CD ∩ EF, and R = EF ∩ AB by solving the parametric equations derived in subtask_2a. "
        "Derive closed-form expressions for the coordinates of P, Q, and R in terms of the hexagon's side length and vector parameters. "
        "Carefully document the algebraic steps and ensure no assumptions are made about the relative positions of these points beyond convexity and parallelism constraints."
    )
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", thinking1.content, answer1.content, thinking2a.content, answer2a.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_2b: {subtask_desc2b}")
    thinking2b, answer2b = await cot_agent_2b([taskInfo, thinking1, answer1, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, computing intersection points, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])

    cot_instruction_2c = (
        "Sub-task 2c: Derive explicit formulas for the side lengths of triangle PQR (i.e., lengths PQ, QR, and RP) in terms of the hexagon's side length and vector parameters. "
        "Avoid simplifying assumptions such as unit vectors or fixed angles. Prepare these formulas for numeric evaluation and validation in the next step."
    )
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_instruction_2c,
        "context": ["user query", thinking1.content, answer1.content, thinking2a.content, answer2a.content, thinking2b.content, answer2b.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_2c: {subtask_desc2c}")
    thinking2c, answer2c = await cot_agent_2c([taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b], cot_instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, deriving triangle side length formulas, thinking: {thinking2c.content}; answer: {answer2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {"thinking": thinking2c, "answer": answer2c}
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])

    debate_instruction_3 = (
        "Sub-task 3: Conduct a Debate-style collaboration where multiple agents propose, critique, and validate the derived formulas for the triangle side lengths from subtask_2c. "
        "Perform symbolic and numeric sanity checks, including testing the formulas on a known special case (e.g., a regular hexagon with 120° angles) to verify correctness. "
        "Identify and resolve any contradictions or invalid assumptions. Ensure that the formulas are dimensionally consistent and geometrically feasible. "
        "Only after consensus and validation proceed to parameter inference."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2a.content, answer2a.content, thinking2b.content, answer2b.content, thinking2c.content, answer2c.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before subtask_3: {subtask_desc3}")
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating triangle side length formulas, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Final decision on validated triangle side length formulas." , is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing validated formulas, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4a = (
        "Sub-task 4a: Using the validated formulas from subtask_3, express the triangle side lengths (200, 240, 300) as equations involving the hexagon's side length and the unknown vector parameters (e.g., angles between vectors, scaling factors). "
        "Avoid assuming unit vectors; instead, allow vector magnitudes and directions to vary within the constraints. Clearly state all equations and unknowns."
    )
    cot_sc_instruction_4b = (
        "Sub-task 4b: Incorporate the closure condition u + v + w = 0 (where u, v, w are vectors representing three consecutive sides) and the parallelism constraints to derive relations between vector magnitudes and angles. "
        "Formulate a solvable system of equations combining these geometric constraints with the triangle side length equations from subtask_4a. Avoid assuming equal angles or magnitudes unless justified by the system."
    )
    cot_sc_instruction_4c = (
        "Sub-task 4c: Solve the system of equations derived in subtask_4b for the hexagon's side length and vector parameters. Use symbolic or numeric methods as appropriate. "
        "During solving, enforce geometric feasibility by checking that all cosine values lie within [-1,1] and that the closure condition holds exactly. "
        "If contradictions arise, revisit previous subtasks for correction. Document all solution steps and justify the chosen root if multiple solutions exist."
    )

    N_sc_4 = self.max_sc
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    cot_agents_4c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]

    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_4a: {subtask_desc4a}")
    for i in range(N_sc_4):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, expressing triangle side lengths as equations, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    best_answer_4a = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[best_answer_4a]
    answer4a = answermapping_4a[best_answer_4a]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query", thinking4a.content, answer4a.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_4b: {subtask_desc4b}")
    for i in range(N_sc_4):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking4a, answer4a], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, incorporating closure and parallelism constraints, thinking: {thinking4b.content}; answer: {answer4b.content}")
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

    possible_answers_4c = []
    thinkingmapping_4c = {}
    answermapping_4c = {}
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_sc_instruction_4c,
        "context": ["user query", thinking4b.content, answer4b.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_4c: {subtask_desc4c}")
    for i in range(N_sc_4):
        thinking4c, answer4c = await cot_agents_4c[i]([taskInfo, thinking4b, answer4b], cot_sc_instruction_4c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4c[i].id}, solving system for hexagon side length, thinking: {thinking4c.content}; answer: {answer4c.content}")
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

    reflect_inst_5 = (
        "Sub-task 5: Verify the computed hexagon side length by substituting it back into the vector and triangle side length relations. "
        "Confirm that all problem conditions are satisfied, including convexity, equilateral sides, parallel opposite sides, and the given triangle side lengths (200, 240, 300). "
        "Perform numeric sanity checks and geometric feasibility tests. Provide a final answer for the hexagon's side length along with a concise justification of correctness. "
        "If verification fails, trigger a rollback to subtask_4c or earlier as needed."
    )
    cot_reflect_instruction_5 = "Sub-task 5: Your problem is to verify and finalize the hexagon side length." + reflect_inst_5
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4c, answer4c]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", thinking4c.content, answer4c.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before subtask_5: {subtask_desc5}")
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying solution, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5],
                                               "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
