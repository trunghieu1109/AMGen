async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: select_and_verify_elements_under_constraints
    cot_instruction_1 = (
        "Sub-task 1: Identify and verify the geometric elements and constraints for points A, B, C, D on the hyperbola x^2/20 - y^2/24 = 1, "
        "such that ABCD is a rhombus with diagonals intersecting at the origin. Confirm symmetry, midpoint, and rhombus properties, "
        "and clarify assumptions about labeling and orientation. Exclude degenerate or non-convex cases."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing geometric constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: derive_and_validate_representations
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the geometric constraints from Sub-task 1, derive algebraic representations for points A, B, C, D on the hyperbola, "
        "expressing A and C as (x1, y1) and (-x1, -y1), B and D as (x2, y2) and (-x2, -y2). Formulate the hyperbola equations, rhombus side length equality, "
        "and perpendicularity of diagonals. Translate these into a system of algebraic equations. Consider multiple independent reasoning attempts for robustness."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, deriving algebraic representations, thinking: {thinking2.content}; answer: {answer2.content}")
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

    # Stage 3: infer_compute_parameters_from_composite_data
    cot_sc_instruction_3 = (
        "Sub-task 3: Analyze and solve the algebraic system from Sub-task 2 to express BD^2 in terms of parameters. "
        "Use hyperbola equations and rhombus conditions to relate variables, simplify expressions, and parameterize if needed. "
        "Prepare the expression for optimization. Consider multiple reasoning attempts for accuracy."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, solving algebraic system, thinking: {thinking3.content}; answer: {answer3.content}")
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

    cot_instruction_4 = (
        "Sub-task 4: Optimize the expression for BD^2 from Sub-task 3 to find its supremum over all valid rhombi inscribed on the hyperbola with diagonals intersecting at the origin. "
        "Use calculus or inequality analysis to determine the maximum possible BD^2, and justify why the supremum is approached but not attained."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, optimizing BD^2, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 4: decompose_simplify_and_sum_components
    debate_instr_5 = (
        "Sub-task 5: Given the optimized expression and reasoning from Sub-task 4, decompose and simplify the expression for BD^2 to a closed-form or numerical value representing the supremum. "
        "Verify consistency with original constraints and provide a clear final answer with justification."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instr_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instr_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instr_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_instr_6 = (
        "Sub-task 6: Review and reflect on the entire solution process and final answers from the debate in Sub-task 5. "
        "Confirm all assumptions and constraints are respected, check for edge cases or degenerate configurations, and ensure robustness and correctness. "
        "If discrepancies arise, refine previous steps accordingly. Provide the final verified answer."
    )
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking5[-1] + all_answer5[-1], final_instr_6, is_sub_task=True)
    agents.append(f"Final Decision agent, reviewing and reflecting, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": final_instr_6,
        "context": ["user query"] + all_thinking5[-1] + all_answer5[-1],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
