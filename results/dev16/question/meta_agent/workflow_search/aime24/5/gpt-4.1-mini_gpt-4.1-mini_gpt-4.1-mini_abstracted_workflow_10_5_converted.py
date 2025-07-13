async def forward_5(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Extract and analyze given edge lengths and geometric properties
    cot_instruction_1_1 = (
        "Sub-task 1: Identify and clearly state all given edge lengths of tetrahedron ABCD, "
        "explicitly listing each edge and its length, and noting the pairs of equal edges. "
        "Emphasize the importance of these lengths as the sole geometric data defining the tetrahedron, "
        "and avoid any assumptions about coordinates or angles at this stage."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, identifying edges, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Analyze the geometric implications of the given equal edge pairs, "
        "deducing any symmetry or special properties of the tetrahedron that can simplify further calculations. "
        "Avoid assuming coordinate placements; instead, focus on intrinsic properties such as edge congruences and their impact on face congruences or tetrahedron classification."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, analyzing edge symmetries, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_1_3 = (
        "Sub-task 3: Define the point I inside the tetrahedron as the incenter, the unique point equidistant from all four faces, "
        "and clarify that the distance sought is the inradius. Explain the geometric significance of the inradius and its relation to volume and surface area, "
        "setting the stage for subsequent calculations."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_1_3,
        "context": ["user query", thinking_1_1.content, thinking_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_1_1, thinking_1_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, defining incenter and inradius, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Compute volume and face areas symbolically, verify consistency
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Compute the volume of tetrahedron ABCD directly from the six given edge lengths using the Cayley–Menger determinant formula. "
        "Perform all calculations symbolically to obtain an exact radical expression for the volume, avoiding any numerical approximations. "
        "Document each step clearly and verify that the volume is positive and consistent with the given edges."
    )
    N_sc = self.max_sc
    cot_sc_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_sc_agents_2_1[i]([taskInfo, thinking_1_1], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_1[i].id}, computing volume, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)

    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_thinkings_2_1, "Sub-task 1: Synthesize and choose the most consistent and correct volume expression for the tetrahedron using Cayley–Menger determinant.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Calculate the area of each of the four faces of tetrahedron ABCD using Heron's formula applied symbolically to the three edges of each face. "
        "Express each face area exactly in simplified radical form, avoiding decimal approximations. "
        "Verify that each area is positive and consistent with the corresponding edges."
    )
    cot_sc_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_2, answer_2_2 = await cot_sc_agents_2_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_2[i].id}, computing face areas, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
        possible_answers_2_2.append(answer_2_2)
        possible_thinkings_2_2.append(thinking_2_2)

    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_thinkings_2_2, "Sub-task 2: Synthesize and choose the most consistent and correct face areas expressions using Heron's formula.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction_2_3 = (
        "Sub-task 3: Sum the four exact face areas obtained from Heron's formula to find the total surface area of the tetrahedron. "
        "Maintain symbolic precision and simplify the resulting expression fully. "
        "Verify that the total surface area is positive and consistent with the individual face areas."
    )
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_reflect_instruction_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3([taskInfo, thinking_2_2], cot_reflect_instruction_2_3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, summing face areas, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 6: ", sub_tasks[-1])

    debate_instruction_2_4 = (
        "Sub-task 4: Verify the consistency and validity of the computed volume and total surface area by checking geometric constraints such as positivity and compatibility with the given edge lengths. "
        "If inconsistencies or contradictions are detected, flag them for re-evaluation before proceeding. "
        "This verification step is critical to prevent error propagation in subsequent calculations. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_4 = self.max_round
    all_thinking_2_4 = [[] for _ in range(N_max_2_4)]
    all_answer_2_4 = [[] for _ in range(N_max_2_4)]
    subtask_desc_2_4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": debate_instruction_2_4,
        "context": ["user query", thinking_2_1.content, thinking_2_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_4):
        for i, agent in enumerate(debate_agents_2_4):
            if r == 0:
                thinking_2_4, answer_2_4 = await agent([taskInfo, thinking_2_1, thinking_2_3], debate_instruction_2_4, r, is_sub_task=True)
            else:
                input_infos_2_4 = [taskInfo, thinking_2_1, thinking_2_3] + all_thinking_2_4[r-1]
                thinking_2_4, answer_2_4 = await agent(input_infos_2_4, debate_instruction_2_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying volume and surface area, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
            all_thinking_2_4[r].append(thinking_2_4)
            all_answer_2_4[r].append(answer_2_4)
    final_decision_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_4, answer_2_4 = await final_decision_agent_2_4([taskInfo] + all_thinking_2_4[-1], "Sub-task 4: Final verification and consistency check of volume and surface area.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking_2_4, "answer": answer_2_4}
    logs.append(subtask_desc_2_4)
    print("Step 7: ", sub_tasks[-1])

    # Stage 3: Calculate and simplify inradius, output final sum
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Calculate the inradius of tetrahedron ABCD using the exact formula r = 3 * volume / total surface area, "
        "substituting the symbolic expressions obtained previously. Perform all algebraic manipulations symbolically to maintain exactness and avoid rounding errors."
    )
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_3_1, answer_3_1 = await cot_sc_agents_3_1[i]([taskInfo, thinking_2_4], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, calculating inradius, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
        possible_answers_3_1.append(answer_3_1)
        possible_thinkings_3_1.append(thinking_3_1)

    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_thinkings_3_1, "Sub-task 1: Synthesize and choose the most consistent and correct inradius expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 8: ", sub_tasks[-1])

    cot_sc_instruction_3_2 = (
        "Sub-task 2: Simplify the inradius expression to the form (m√n)/p, where m and p are positive integers that are coprime, "
        "and n is a positive square-free integer. Carefully factor and reduce the expression, ensuring no prime square divides n and that m and p share no common factors. "
        "Avoid premature numerical approximations or truncations."
    )
    cot_sc_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_2 = []
    possible_thinkings_3_2 = []
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_sc_instruction_3_2,
        "context": ["user query", thinking_3_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_3_2, answer_3_2 = await cot_sc_agents_3_2[i]([taskInfo, thinking_3_1], cot_sc_instruction_3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_2[i].id}, simplifying inradius, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
        possible_answers_3_2.append(answer_3_2)
        possible_thinkings_3_2.append(thinking_3_2)

    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + possible_thinkings_3_2, "Sub-task 2: Synthesize and choose the most consistent and correct simplified inradius expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_3.subtask_2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 9: ", sub_tasks[-1])

    cot_reflect_instruction_3_3 = (
        "Sub-task 3: Compute and output the sum m + n + p based on the simplified inradius expression. "
        "Confirm that the final answer is consistent with all previous symbolic computations and that the form meets all problem requirements. "
        "If the expression unexpectedly reduces to a rational number or contradicts earlier results, trigger a re-examination of prior steps. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_3 = self.max_round
    cot_inputs_3_3 = [taskInfo, thinking_3_1, thinking_3_2]
    subtask_desc_3_3 = {
        "subtask_id": "stage_3.subtask_3",
        "instruction": cot_reflect_instruction_3_3,
        "context": ["user query", thinking_3_1.content, thinking_3_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_3, answer_3_3 = await cot_agent_3_3(cot_inputs_3_3, cot_reflect_instruction_3_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, computing final sum, thinking: {thinking_3_3.content}; answer: {answer_3_3.content}")
    for i in range(N_max_3_3):
        feedback_3_3, correct_3_3 = await critic_agent_3_3([taskInfo, thinking_3_3], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_3.id}, providing feedback, thinking: {feedback_3_3.content}; answer: {correct_3_3.content}")
        if correct_3_3.content == "True":
            break
        cot_inputs_3_3.extend([thinking_3_3, feedback_3_3])
        thinking_3_3, answer_3_3 = await cot_agent_3_3(cot_inputs_3_3, cot_reflect_instruction_3_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, refining final sum, thinking: {thinking_3_3.content}; answer: {answer_3_3.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_3 output: thinking - {thinking_3_3.content}; answer - {answer_3_3.content}")
    subtask_desc_3_3['response'] = {"thinking": thinking_3_3, "answer": answer_3_3}
    logs.append(subtask_desc_3_3)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_3, answer_3_3, sub_tasks, agents)
    return final_answer, logs
