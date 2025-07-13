async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    # Stage 0: Extract and prepare edge lengths
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and explicitly list all given edge lengths of tetrahedron ABCD, "
        "including the equalities AB = CD = sqrt(41), AC = BD = sqrt(80), and BC = AD = sqrt(89). "
        "Verify the consistency of these equalities and confirm that all six edges are accounted for without contradictions. "
        "Avoid making assumptions about the shape beyond the given data."
    )
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting edge lengths, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Compute and record the squared lengths of all edges identified in stage_0.subtask_1, "
        "ensuring exact symbolic expressions are used (e.g., 41, 80, 89) without decimal approximations. "
        "Prepare these squared lengths for use in subsequent geometric calculations."
    )
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing squared edge lengths, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Compute face areas, validate, sum, compute volume, validate
    N_sc = self.max_sc

    # Subtask 1: Calculate area of each face using Heron's formula with exact squared lengths
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Calculate the area of each of the four triangular faces of tetrahedron ABCD using Heron's formula "
        "with the exact squared edge lengths from stage_0.subtask_2. For each face, express the area in simplest radical form, "
        "carefully factor the radicand to ensure it is square-free, and avoid premature numerical approximations. "
        "Document each step of the simplification explicitly."
    )
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, calculating face areas, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent and correct face areas.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Validate radical expressions for each face area by numeric approximation
    cot_instruction_1_2 = (
        "Sub-task 2: Validate the radical expressions for each face area obtained in subtask_1 by numerically approximating them "
        "to confirm consistency with the symbolic radical forms. Ensure no significant discrepancies exist that would indicate algebraic errors or premature approximations."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, validating face areas, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Subtask 3: Sum the four exact face areas to get total surface area
    cot_sc_instruction_1_3 = (
        "Sub-task 3: Sum the four exact face areas from subtask_1 to obtain the total surface area of the tetrahedron. "
        "Express the total surface area in simplest radical form, ensuring the radical is square-free and the expression is fully simplified. "
        "Avoid using approximate values or placeholder expressions."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, summing face areas, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)

    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize and choose the most consistent total surface area.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Subtask 4: Compute volume using Cayley-Menger determinant with exact squared lengths
    cot_sc_instruction_1_4 = (
        "Sub-task 4: Compute the volume of tetrahedron ABCD using the Cayley-Menger determinant formula with the exact squared edge lengths from stage_0.subtask_2. "
        "Express the volume in simplest radical form, ensuring the radicand is square-free and the expression is fully simplified. "
        "Verify the volume is positive and consistent with the tetrahedron's geometry."
    )
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_4, answer_1_4 = await cot_agents_1_4[i]([taskInfo, thinking_0_2], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, computing volume, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
        possible_answers_1_4.append(answer_1_4)
        possible_thinkings_1_4.append(thinking_1_4)

    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_thinkings_1_4, "Sub-task 4: Synthesize and choose the most consistent volume.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    # Subtask 5: Validate volume expression by numeric approximation
    cot_instruction_1_5 = (
        "Sub-task 5: Validate the volume expression from subtask_4 by numerical approximation and cross-check with the symbolic radical form to confirm accuracy and consistency. "
        "Address any discrepancies by revisiting the calculation or simplification steps."
    )
    cot_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_5 = {
        "subtask_id": "stage_1.subtask_5",
        "instruction": cot_instruction_1_5,
        "context": ["user query", thinking_1_4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_5, answer_1_5 = await cot_agent_1_5([taskInfo, thinking_1_4], cot_instruction_1_5, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_5.id}, validating volume, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    sub_tasks.append(f"Stage 1 Subtask 5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    subtask_desc_1_5['response'] = {"thinking": thinking_1_5, "answer": answer_1_5}
    logs.append(subtask_desc_1_5)
    print("Step 1.5: ", sub_tasks[-1])

    # Stage 2: Calculate and simplify inradius

    # Subtask 1: Calculate inradius r = (3 * volume) / (surface area) with exact radicals
    cot_instruction_2_1 = (
        "Sub-task 1: Calculate the inradius r of the tetrahedron using the formula r = (3 * volume) / (surface area), "
        "substituting the exact radical expressions for volume and surface area from stage_1.subtask_4 and stage_1.subtask_3. "
        "Express the resulting inradius as a fraction involving radicals without any simplification at this stage."
    )
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3.content, thinking_1_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent([taskInfo, thinking_1_3, thinking_1_4], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculating inradius, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Subtask 2: Simplify inradius expression into (m sqrt n)/p form with coprimality and square-freeness checks
    cot_instruction_2_2 = (
        "Sub-task 2: Simplify the inradius expression obtained in subtask_1 into the form (m sqrt n)/p, where m and p are positive integers that are coprime, "
        "and n is a positive square-free integer. Carefully factor numerator and denominator, verify that no invalid cancellations involving irrational terms occur, "
        "and explicitly confirm the irreducibility of the fraction. Document each simplification step and verify the square-freeness of n."
    )
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent([taskInfo, thinking_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, simplifying inradius, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 Subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Subtask 3: Final validation of simplified inradius by numeric approximation and coprimality check
    cot_instruction_2_3 = (
        "Sub-task 3: Perform a final validation of the simplified inradius expression by numerically approximating the value and comparing it with the numeric approximation of the original formula r = (3 * volume) / (surface area). "
        "Confirm that the simplified radical form is consistent and plausible. Also verify that m and p are coprime and n is square-free as per problem requirements."
    )
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_instruction_2_3,
        "context": ["user query", thinking_2_2.content, thinking_1_5.content, thinking_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3([taskInfo, thinking_2_2, thinking_1_5, thinking_1_2], cot_instruction_2_3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, validating simplified inradius, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Stage 2 Subtask 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    # Subtask 4: Compute final sum m + n + p and present final answer
    cot_instruction_2_4 = (
        "Sub-task 4: Compute the sum m + n + p from the simplified inradius expression and present this final result clearly as the answer to the problem. "
        "Ensure that the sum is computed only after all validations are complete."
    )
    subtask_desc_2_4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_instruction_2_4,
        "context": ["user query", thinking_2_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_4, answer_2_4 = await cot_agent([taskInfo, thinking_2_3], cot_instruction_2_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing final sum, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
    sub_tasks.append(f"Stage 2 Subtask 4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking_2_4, "answer": answer_2_4}
    logs.append(subtask_desc_2_4)
    print("Step 2.4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_4, answer_2_4, sub_tasks, agents)
    return final_answer, logs
