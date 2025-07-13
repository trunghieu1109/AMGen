async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: derive_and_validate_representations
    cot_instruction_1 = (
        "Sub-task 1: Derive a consistent geometric representation of tetrahedron ABCD using the given edge lengths: "
        "AB=CD=sqrt(41), AC=BD=sqrt(80), BC=AD=sqrt(89). "
        "Assign coordinates or vectors to vertices, confirm non-degeneracy, and identify symmetries. "
        "Provide explicit vertex coordinates or edge vectors suitable for further calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, deriving geometric representation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)

    cot_sc_instruction_1b = (
        "Sub-task 2: Based on the derived geometric representation, verify consistency and non-degeneracy of the tetrahedron. "
        "Check triangle inequalities and symmetry properties. Confirm or correct vertex assignments."
    )
    N_sc = self.max_sc
    cot_agents_sc_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc_1b = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1b, answer1b = await cot_agents_sc_1b[i]([taskInfo, thinking1, answer1], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc_1b[i].id}, verifying representation, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b
    best_answer_1b = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[best_answer_1b]
    answer1b = answermapping_1b[best_answer_1b]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)

    print("Step 1: ", sub_tasks[-1])

    # Stage 1: derive_composite_measure (compute face areas)
    reflexion_instruction_2 = (
        "Sub-task 3: Using the validated geometric representation, compute the areas of all four triangular faces of tetrahedron ABCD. "
        "Use vector cross products or Heron's formula as appropriate. Verify consistency with given edge lengths. "
        "Provide exact or symbolic expressions for each face area."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_2 = [taskInfo, thinking1b, answer1b]
    subtask_desc_2 = {
        "subtask_id": "subtask_3",
        "instruction": reflexion_instruction_2,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflexion_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, computing face areas, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_reflect):
        feedback2, correct2 = await critic_agent_2([taskInfo, thinking2, answer2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback on face areas, thinking: {feedback2.content}; answer: {correct2.content}")
        if correct2.content == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback2])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflexion_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining face areas, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)

    print("Step 2: ", sub_tasks[-1])

    # Stage 2: infer_compute_parameters_from_composite_data (compute volume)
    cot_instruction_3 = (
        "Sub-task 4: Calculate the volume of tetrahedron ABCD using the vertex coordinates or edge vectors derived earlier. "
        "Use the scalar triple product or Cayley-Menger determinant formula. Confirm volume is positive and consistent with face areas. "
        "Express volume in simplest radical or symbolic form."
    )
    N_sc_3 = self.max_sc
    cot_agents_sc_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3):
        thinking3, answer3 = await cot_agents_sc_3[i]([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc_3[i].id}, calculating volume, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    best_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[best_answer_3]
    answer3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)

    print("Step 3: ", sub_tasks[-1])

    # Stage 3: select_and_verify_elements_under_constraints (compute inradius and verify)
    cot_instruction_4 = (
        "Sub-task 5: Using the volume and face areas, compute the inradius r = 3 * Volume / (sum of face areas). "
        "Simplify r to the form (m * sqrt(n)) / p with m, n, p positive integers, m and p coprime, n square-free. "
        "Verify correctness by cross-checking geometric properties and provide final sum m + n + p."
    )
    N_sc_4 = self.max_sc
    cot_agents_sc_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc_4 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4):
        thinking4, answer4 = await cot_agents_sc_4[i]([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc_4[i].id}, computing inradius and verifying, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    best_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[best_answer_4]
    answer4 = answermapping_4[best_answer_4]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)

    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
