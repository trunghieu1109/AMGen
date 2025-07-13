async def forward_1(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Establish a coordinate system and assign exact coordinates to points A, B, and C such that AB=5, BC=9, and AC=10. "
        "Use symbolic or exact arithmetic to find coordinates of A, B, C and derive the exact equation of the circumcircle ω passing through these points. "
        "Output the coordinates of A, B, C and the circle equation in a structured format (e.g., JSON)."
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, determining coordinates and circle, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    best_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[best_answer_1]
    answer1 = answermapping_1[best_answer_1]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Using the exact coordinates of B and C and the circle equation ω from Sub-task 1, find the exact equations of the tangent lines to ω at B and C. "
        "Compute their intersection point D symbolically or with exact arithmetic. Verify D lies outside ω. Output coordinates of D in structured format."
    )
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, finding tangent lines and point D, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3a = (
        "Sub-task 3a: Using exact coordinates of A and D from previous subtasks, write the parametric equation of line AD symbolically. "
        "Express line AD as (x,y) = A + t*(D - A) with parameter t. Output the parametric form explicitly."
    )
    cot_sc_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3a, answer3a = await cot_sc_agents_3a[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3a, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3a[i].id}, formulating parametric line AD, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    best_answer_3a = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[best_answer_3a]
    answer3a = answermapping_3a[best_answer_3a]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_sc_instruction_3b = (
        "Sub-task 3b: Substitute the parametric form of line AD from Sub-task 3a into the exact circle equation ω from Sub-task 1. "
        "Solve the resulting quadratic equation symbolically for parameter t. Identify the root t=0 corresponding to point A and select the other root for point P. "
        "Compute exact coordinates of P and verify P lies on ω and is distinct from A. Output coordinates of P in structured format."
    )
    cot_sc_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", thinking3a.content, answer3a.content, thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3b, answer3b = await cot_sc_agents_3b[i]([taskInfo, thinking3a, answer3a, thinking1, answer1], cot_sc_instruction_3b, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3b[i].id}, solving quadratic for P, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    best_answer_3b = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[best_answer_3b]
    answer3b = answermapping_3b[best_answer_3b]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Calculate the length AP using exact coordinates of A and P from Sub-task 3b. "
        "Express AP as a reduced fraction m/n with relatively prime integers m and n. "
        "Compute and output the sum m + n as the final numeric answer. Verify fraction is fully simplified."
    )
    cot_sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3b.content, answer3b.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking4, answer4 = await cot_sc_agents_4[i]([taskInfo, thinking3b, answer3b], cot_sc_instruction_4, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, calculating length AP and final answer, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    best_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[best_answer_4]
    answer4 = answermapping_4[best_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_reflect_instruction_5 = (
        "Sub-task 5: Review all previous subtasks' outputs for consistency, correctness, and coordinate system coherence. "
        "Verify all distances and points satisfy the problem's geometric constraints exactly. Confirm the final answer m + n matches the computed length AP and no rounding or assumption errors occurred. "
        "If inconsistencies are found, flag them for correction; otherwise, finalize and return the verified answer."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3a, answer3a, thinking3b, answer3b, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying all subtasks, thinking: {thinking5.content}; answer: {answer5.content}")
    critic_inst_5 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5(cot_inputs_5 + [thinking5, answer5], "Please review and provide the limitations of provided solutions. " + critic_inst_5, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
