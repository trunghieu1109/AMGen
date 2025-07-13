async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Verify the existence of tetrahedron ABCD with edges AB=CD=sqrt(41), AC=BD=sqrt(80), BC=AD=sqrt(89). "
        "Compute the volume V using the Cayley-Menger determinant formula. "
        "Set up the distance matrix from the given edges, calculate the determinant, and extract the volume. "
        "Ensure the volume is positive and the tetrahedron is non-degenerate. "
        "Provide the exact symbolic expression for V without decimal approximations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, computing volume, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Using the volume V from Sub-task 1, compute the areas of the four triangular faces ABC, ABD, ACD, and BCD. "
        "Assign edges carefully: for ABC use AB, BC, AC; for ABD use AB, BD, AD; for ACD use AC, CD, AD; for BCD use BC, CD, BD. "
        "Use Heron's formula to find each area exactly, then sum to get total surface area S. "
        "Avoid decimal approximations until final step."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
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
        "Sub-task 3: Calculate the inradius r of the tetrahedron using r = 3V / S, where V is volume from Sub-task 1 and S is total surface area from Sub-task 2. "
        "Express r in simplest radical form m*sqrt(n)/p, with m, n, p positive integers, m and p coprime, and n square-free. "
        "Simplify radicals and fractions exactly, avoiding decimal approximations."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculating inradius, thinking: {thinking3.content}; answer: {answer3.content}")
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

    reflect_inst = "Given previous calculations, verify the simplification and correctness of the inradius expression and the sum m+n+p. Refine if necessary."
    cot_reflect_instruction_4 = (
        "Sub-task 4: Using the simplified inradius expression from Sub-task 3, extract integers m, n, p as defined. "
        "Compute the sum m + n + p. Verify all previous steps for consistency and correctness. "
        + reflect_inst
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, verifying and refining final answer, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining final answer, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
