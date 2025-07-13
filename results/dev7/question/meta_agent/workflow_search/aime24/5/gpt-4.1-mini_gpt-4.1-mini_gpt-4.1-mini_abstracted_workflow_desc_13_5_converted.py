async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    N_sc = self.max_sc
    # Stage 0 - derive_and_validate_representations
    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent tetrahedron ABCD with given edge lengths AB=CD=sqrt(41), AC=BD=sqrt(80), BC=AD=sqrt(89). "
        "Derive exact symbolic coordinates or vector framework consistent with these lengths, ensuring non-degeneracy and triangle inequalities symbolically. "
        "Avoid decimal approximations. Provide detailed reasoning and final symbolic representation."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving symbolic representation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Using the symbolic representation from Sub-task 1, compute the squared volume V^2 of tetrahedron ABCD exactly via the Cayley–Menger determinant formula. "
        "Simplify the radical expression fully, ensuring all radicands are square-free and expressions remain exact. "
        "Avoid coordinate-based volume calculations and decimal approximations. Provide detailed symbolic steps and final simplified V^2."
    )
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, computing exact volume squared, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent exact volume squared expression.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Calculate exact areas of the four triangular faces of tetrahedron ABCD using Heron's formula symbolically with given edge lengths. "
        "Compute semi-perimeters and area squared for each face, simplify radicals fully, and ensure square-free radicands. "
        "Avoid decimal approximations. Confirm consistency with edge lengths and provide detailed symbolic calculations and final areas."
    )
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0_subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, computing exact face areas, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_answers_0_3 + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent exact face areas.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Stage 1 - derive_composite_measure
    reflexion_instruction_1_1 = (
        "Sub-task 1: Sum the exact symbolic areas of the four faces from Stage 0 Sub-task 3 to compute total surface area S. "
        "Maintain symbolic form and simplify without approximations. Review and refine calculation for accuracy and consistency using reflexion."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": reflexion_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3], reflexion_instruction_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, calculating total surface area, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_1_1([taskInfo, thinking_1_1, answer_1_1], "Please review and provide limitations of the surface area calculation. If correct, output exactly 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3, thinking_1_1, answer_1_1, feedback], reflexion_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining surface area, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    reflexion_instruction_1_2 = (
        "Sub-task 2: Derive the exact volume V by taking the positive square root of V^2 from Stage 0 Sub-task 2 symbolically. "
        "Simplify radicals fully, maintain exact symbolic form, and review/refine calculation for accuracy and consistency using reflexion."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": reflexion_instruction_1_2,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_0_2, answer_0_2], reflexion_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, calculating exact volume, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_1_2([taskInfo, thinking_1_2, answer_1_2], "Please review and provide limitations of the volume calculation. If correct, output exactly 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_0_2, answer_0_2, thinking_1_2, answer_1_2, feedback], reflexion_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining volume, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Stage 2 - infer_compute_parameters_from_composite_data
    cot_instruction_2_1 = (
        "Sub-task 1: Compute the inradius r = 3 * V / S using exact symbolic volume V and surface area S from Stage 1. "
        "Perform symbolic division and simplify the radical expression fully, preparing for final simplification into form (m√n)/p. "
        "Provide detailed reasoning and intermediate steps without decimal approximations."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing symbolic inradius, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Simplify the inradius expression from Sub-task 1 to canonical form (m√n)/p, where m, n, p are positive integers, m and p coprime, n square-free. "
        "Explicitly identify m, n, p and document prime factorization and radical simplification steps. "
        "Use self-consistency by generating multiple independent simplifications and choosing the most consistent result."
    )
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, simplifying inradius, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Choose the most consistent simplified inradius form.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    # Stage 3 - select_and_verify_elements_under_constraints
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Perform numeric consistency check by evaluating exact symbolic inradius (m√n)/p and numeric value of 3V/S from exact volume and surface area to high precision. "
        "Confirm agreement within negligible tolerance and verify point I lies inside tetrahedron. Document verification results explicitly. "
        "Use self-consistency and reflexion collaboration."
    )
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content, thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_3_1[i]([taskInfo, thinking_2_2, answer_2_2, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, verifying inradius numerically, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_answers_3_1 + possible_thinkings_3_1, "Sub-task 3: Confirm correctness of inradius and numeric consistency.", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    cot_instruction_3_2 = (
        "Sub-task 2: Compute final answer m + n + p using values identified in Stage 2 Sub-task 2. "
        "Present final answer clearly with brief summary and verification outcomes."
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_2 = {
        "subtask_id": "stage_3_subtask_2",
        "instruction": cot_instruction_3_2,
        "context": ["user query", thinking_2_2.content, answer_2_2.content, thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2([taskInfo, thinking_2_2, answer_2_2, thinking_3_1, answer_3_1], cot_instruction_3_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_2.id}, computing final sum m+n+p, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Stage 3 Sub-task 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
