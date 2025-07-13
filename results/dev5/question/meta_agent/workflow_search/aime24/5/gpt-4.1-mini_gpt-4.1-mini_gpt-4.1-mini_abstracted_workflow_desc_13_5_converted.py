async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 0.1: Derive a consistent geometric representation of tetrahedron ABCD using the given edge lengths: "
        "AB=CD=\u221A41, AC=BD=\u221A80, BC=AD=\u221A89. Identify symmetries and set up exact coordinate or vector relations "
        "to represent vertices A, B, C, D symbolically without unjustified assumptions. Document all symbolic constraints."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving geometric representation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 0.2: Validate the geometric representation derived in Sub-task 0.1 by verifying all given edge length equalities symbolically and numerically. "
        "Confirm the tetrahedron is non-degenerate with positive volume and that the incenter exists inside. Resolve any inconsistencies before proceeding."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, validating representation, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 0.2: Synthesize and choose the most consistent validation of the geometric representation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    print("Step 0: ", sub_tasks[-1])

    cot_instruction_1_1 = (
        "Sub-task 1.1: Compute exact symbolic expressions for the areas of the four faces of tetrahedron ABCD using the validated coordinates or vector representations. "
        "Use vector cross products of edge vectors to find each triangular face area exactly, maintaining radicals and rational numbers. Document each face area explicitly."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, computing face areas, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_instruction_1_2 = (
        "Sub-task 1.2: Calculate the exact symbolic volume of tetrahedron ABCD using the vertex coordinates or edge lengths. "
        "Apply the scalar triple product or Cayley-Menger determinant formula to obtain a precise symbolic expression for the volume. "
        "Reflect on the calculation to ensure accuracy and consistency with face areas."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_2 = self.max_round
    cot_inputs_1_2 = [taskInfo, thinking_0_2, answer_0_2]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, calculating volume, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    for i in range(N_max_1_2):
        feedback, correct = await critic_agent_1_2([taskInfo, thinking_1_2, answer_1_2], "Please review and provide the limitations of the volume calculation. If correct, output exactly 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_2.extend([thinking_1_2, answer_1_2, feedback])
        thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining volume calculation, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 2.1: Perform a dedicated consistency check between the computed symbolic face areas and volume. "
        "Verify volume positivity and that the sum of face areas is consistent with the tetrahedron's geometry. "
        "Cross-validate numeric approximations derived from symbolic expressions to detect discrepancies. "
        "If inconsistencies arise, initiate reflexion or debate to reconcile differences before proceeding."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, temperature=0.5, role=role) for role in self.debate_role]
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion | Debate"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, consistency checking, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback, correct = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], "Please review and provide limitations of the consistency check. If correct, output exactly 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining consistency check, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    if answer_2_1.content != "True":
        for r in range(N_max_2_1):
            all_thinking = []
            all_answer = []
            for agent in debate_agents_2_1:
                if r == 0:
                    thinking_d, answer_d = await agent([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_1, r, is_sub_task=True)
                else:
                    input_infos = [taskInfo, thinking_2_1, answer_2_1] + all_thinking + all_answer
                    thinking_d, answer_d = await agent(input_infos, cot_instruction_2_1, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, consistency debate, thinking: {thinking_d.content}; answer: {answer_d.content}")
                all_thinking.append(thinking_d)
                all_answer.append(answer_d)
        final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking + all_answer, "Sub-task 2.3: Synthesize and finalize consistency check.", is_sub_task=True)
        agents.append(f"Final Decision agent, consistency check finalizing, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.3 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3_1 = (
        "Sub-task 3.1: Derive the exact symbolic expression for the inradius of tetrahedron ABCD using the formula inradius = (3 × volume) / (sum of face areas). "
        "Substitute the previously computed exact symbolic volume and face areas. Simplify carefully, maintaining radicals in simplest form and rational factors reduced."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_1, answer_2_1], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, deriving inradius, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    cot_sc_instruction_3_2 = (
        "Sub-task 3.2: Simplify the inradius expression into the canonical form (m * sqrt(n)) / p, where m, n, p are positive integers, "
        "m and p are coprime, and n is square-free. Perform factorization, rationalization, and verify the square-free condition for n. Document simplification steps explicitly."
    )
    N_sc_3_2 = self.max_sc
    cot_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_2)]
    possible_answers_3_2 = []
    possible_thinkings_3_2 = []
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_sc_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3_2):
        thinking_i, answer_i = await cot_agents_3_2[i]([taskInfo, thinking_3_1, answer_3_1], cot_sc_instruction_3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_2[i].id}, simplifying inradius, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_2.append(answer_i)
        possible_thinkings_3_2.append(thinking_i)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + possible_answers_3_2 + possible_thinkings_3_2, "Sub-task 3.2: Synthesize and choose the most consistent simplified inradius expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)

    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4_1 = (
        "Sub-task 4.1: Verify the correctness and geometric plausibility of the simplified inradius expression. "
        "Cross-check the symbolic result against numeric approximations from original coordinates and edge lengths. "
        "Confirm positivity and that point I lies inside the tetrahedron. Address discrepancies through reflexion or debate before finalizing."
    )
    N_sc_4_1 = self.max_sc
    cot_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4_1)]
    possible_answers_4_1 = []
    possible_thinkings_4_1 = []
    subtask_desc_4_1 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": cot_sc_instruction_4_1,
        "context": ["user query", thinking_3_2.content, answer_3_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4_1):
        thinking_i, answer_i = await cot_agents_4_1[i]([taskInfo, thinking_3_2, answer_3_2], cot_sc_instruction_4_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4_1[i].id}, verifying inradius, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_4_1.append(answer_i)
        possible_thinkings_4_1.append(thinking_i)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4_1, answer_4_1 = await final_decision_agent_4_1([taskInfo] + possible_answers_4_1 + possible_thinkings_4_1, "Sub-task 4.1: Synthesize and confirm the correctness of the inradius verification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking_4_1.content}; answer - {answer_4_1.content}")
    subtask_desc_4_1['response'] = {"thinking": thinking_4_1, "answer": answer_4_1}
    logs.append(subtask_desc_4_1)

    cot_instruction_4_2 = (
        "Sub-task 4.2: Compute the final answer m + n + p from the simplified inradius expression (m * sqrt(n)) / p. "
        "Present a clear, concise summary of the solution, including reasoning steps, verification results, and the final numeric sum."
    )
    cot_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4_2 = {
        "subtask_id": "stage_4.subtask_2",
        "instruction": cot_instruction_4_2,
        "context": ["user query", thinking_3_2.content, answer_3_2.content, thinking_4_1.content, answer_4_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_4_2, answer_4_2 = await cot_agent_4_2([taskInfo, thinking_3_2, answer_3_2, thinking_4_1, answer_4_1], cot_instruction_4_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4_2.id}, computing final sum, thinking: {thinking_4_2.content}; answer: {answer_4_2.content}")
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking_4_2.content}; answer - {answer_4_2.content}")
    subtask_desc_4_2['response'] = {"thinking": thinking_4_2, "answer": answer_4_2}
    logs.append(subtask_desc_4_2)

    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4_2, answer_4_2, sub_tasks, agents)
    return final_answer, logs
