async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Derive a comprehensive vector representation of the convex equilateral hexagon ABCDEF with opposite sides parallel. "
        "Assign vectors to sides AB, BC, CD, DE, EF, and FA, ensuring that opposite sides are parallel and equal in length. "
        "Express these parallelism conditions as vector equalities or scalar multiples. Represent the hexagon as a closed vector polygon (sum of side vectors equals zero). "
        "Define angle variables (e.g., alpha, beta) to parameterize directions of vectors AB, BC, and CD, and express the remaining sides accordingly. "
        "Avoid premature coordinate assignments but establish symbolic vector forms with clear orientation and convexity assumptions. "
        "This subtask must produce explicit symbolic vector expressions and angle definitions to be used downstream."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, deriving vector representation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Using the vector representation from Sub-task 1, derive explicit parametric expressions for the intersection points P, Q, and R formed by the extended lines AB & CD, CD & EF, and EF & AB respectively. "
        "Set up and solve the parametric line equations symbolically to express P, Q, and R in terms of the hexagon side length s and angle variables (alpha, beta). "
        "Ensure these expressions are closed-form and suitable for algebraic manipulation. Emphasize the geometric meaning of parameters and maintain consistency with convexity and orientation assumptions. "
        "Avoid numeric approximations or assumptions about parameter signs at this stage."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, deriving parametric intersection points, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Using the parametric expressions for points P, Q, and R from Sub-task 2, express the side lengths |PQ|, |QR|, and |RP| of the triangle formed by these points as explicit symbolic functions of the hexagon side length s and angle variables (alpha, beta). "
        "Derive vector differences representing these sides, then compute their magnitudes symbolically. Present these formulas clearly, ensuring they can be numerically evaluated once s and angles are known. "
        "Avoid simplifying assumptions that bypass the parametric dependencies."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, expressing triangle side lengths symbolically, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Set up the system of equations equating the symbolic expressions for |PQ|, |QR|, and |RP| from Sub-task 3 to the given triangle side lengths 200, 240, and 300 respectively. "
        "Solve this nonlinear system for the unknown hexagon side length s and angle variables (alpha, beta). Break down the solving process into: (1) isolating parameters, (2) applying numeric root-finding or symbolic solving methods, and (3) verifying parameter feasibility (positivity, convexity constraints). "
        "Avoid arbitrary angle assumptions or oversimplified sine-law applications. Document the solving steps and justify choices of numeric methods or approximations. "
        "Use self-consistency by generating multiple solution attempts and selecting the most consistent answer."
    )
    N = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, solving nonlinear system for s, alpha, beta, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    best_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[best_answer_4]
    answer4 = answermapping_4[best_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflect_inst_5 = (
        "Sub-task 5: Perform a rigorous numeric verification of the solution obtained in Sub-task 4. "
        "Substitute the found values of s, alpha, and beta back into the parametric expressions for P, Q, and R, then compute the triangle side lengths |PQ|, |QR|, and |RP| numerically. "
        "Confirm that these lengths match the given values (200, 240, 300) within a reasonable tolerance. Additionally, verify that the hexagon side vectors satisfy the closure condition and that the hexagon remains convex with all sides equal and opposite sides parallel. "
        "Identify and discuss any discrepancies or inconsistencies. This verification step is critical to catch errors early and ensure solution validity. "
        "Use reflexion to iteratively refine the solution if needed."
    )
    cot_reflect_instruction_5 = "Sub-task 5: Your problem is to verify and refine the numeric solution." + reflect_inst_5
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, thinking4.content, answer4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying numeric solution, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5],
                                               "Please review and provide the limitations of the provided solution. If correct, output exactly 'True' in 'correct'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    reflect_inst_6 = (
        "Sub-task 6: Synthesize the results from the solving and verification subtasks to produce the final answer: the side length s of the hexagon. "
        "Provide a clear, justified conclusion including the numeric value of s, a summary of the verification results, and any assumptions or approximations made. "
        "Discuss the geometric interpretation of the solution and confirm that all problem constraints are satisfied. If inconsistencies remain, propose possible reasons and suggest further refinement steps. "
        "This subtask ensures the solution is complete, transparent, and reliable."
    )
    cot_reflect_instruction_6 = "Sub-task 6: Your problem is to finalize and justify the hexagon side length solution." + reflect_inst_6
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, thinking4.content, answer4.content, thinking5.content, answer5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, synthesizing final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
