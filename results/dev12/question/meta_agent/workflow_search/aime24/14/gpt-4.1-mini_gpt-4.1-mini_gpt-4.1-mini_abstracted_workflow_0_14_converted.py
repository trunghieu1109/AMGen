async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # --------------------------------------------------------------------------------------------------------------
    # Stage 0: Formal Representation and Constraint Formulation
    # Subtask 1: Parameterize hyperbola and points A,B,C,D
    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent the hyperbola equation x^2/20 - y^2/24 = 1 and introduce parameters "
        "to describe points A, B, C, D on the hyperbola. Ensure symmetric parameterization about the origin "
        "suitable for algebraic manipulation without premature assumptions."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, parameterize hyperbola and points, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Subtask 2: Express midpoint condition of diagonals intersecting at origin
    cot_instruction_0_2 = (
        "Sub-task 2: Express algebraically the condition that the diagonals of the rhombus intersect at the origin, "
        "implying the origin is the midpoint of both diagonals AC and BD. Represent this explicitly in terms of parameters."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, express midpoint condition, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Subtask 3: Formulate rhombus properties as algebraic conditions (equal sides, perpendicular diagonals)
    cot_instruction_0_3 = (
        "Sub-task 3: Formulate the rhombus properties as algebraic conditions: equal side lengths and perpendicular diagonals. "
        "Translate these geometric conditions into equations involving coordinates or parameters of A, B, C, D, treating them as a coupled system."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, formulate rhombus conditions, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Subtask 4: Summarize all constraints into a single coherent system
    cot_instruction_0_4 = (
        "Sub-task 4: Summarize all constraints (hyperbola equations, midpoint conditions, rhombus side equality, diagonal perpendicularity) "
        "into a single coherent system of equations or relations, highlighting parameter coupling and preparing for elimination."
    )
    cot_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": cot_instruction_0_4,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_4, answer_0_4 = await cot_agent_0_4([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3], cot_instruction_0_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_4.id}, summarize constraints, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    sub_tasks.append(f"Sub-task 0.4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)

    # --------------------------------------------------------------------------------------------------------------
    # Stage 1: Parameterization and Constraint Reduction
    # Subtask 1: Parameterize A and C using hyperbolic functions, express B and D accordingly
    cot_instruction_1_1 = (
        "Sub-task 1: Parameterize points A and C on the hyperbola using hyperbolic functions (a cosh u, b sinh u), "
        "express points B and D using midpoint and symmetry conditions, respecting domain and symmetry."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_4, answer_0_4], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, parameterize points A,C and express B,D, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Subtask 2: Derive single independent constraint coupling parameters u,v and express u in terms of v with domain restrictions
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Derive the single independent constraint coupling parameters u and v (e.g., tanh u * tanh v = -a^2/b^2). "
        "Express u in terms of v (u = artanh(-a^2/(b^2 tanh v))) and analyze domain restrictions ensuring real, valid, nondegenerate rhombi."
    )
    N_sc = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, derive constraint and domain restrictions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 1.2: Synthesize and choose the most consistent constraint and domain analysis.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize constraint and domain, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Subtask 3: Express BD^2 purely as function of single parameter v by substituting u(v)
    cot_instruction_1_3 = (
        "Sub-task 3: Express BD^2 purely as a function of the single parameter v by substituting the relation u = artanh(-a^2/(b^2 tanh v)). "
        "Simplify the expression for BD^2 suitable for optimization, avoiding premature conclusions."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_1_2, answer_1_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, express BD^2 as function of v, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # --------------------------------------------------------------------------------------------------------------
    # Stage 2: Feasibility Analysis and Optimization
    # Subtask 1: Analyze feasibility and nondegeneracy of candidate parameters
    cot_reflect_instruction_2_1 = (
        "Sub-task 2.1: Analyze the feasibility and nondegeneracy of candidate parameter values from the constraint and BD^2 expression. "
        "Verify parameters are real, nonzero, and produce valid, convex rhombi on the hyperbola. Exclude degenerate/boundary cases. "
        "Document feasible parameter domain for optimization."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_2, answer_1_2, thinking_1_3, answer_1_3]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content, thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, analyze feasibility and nondegeneracy, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], "Please review and provide limitations of the solution. If correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback on feasibility, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refine feasibility analysis, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Subtask 2: Perform constrained optimization of BD^2 over feasible domain
    cot_reflect_instruction_2_2 = (
        "Sub-task 2.2: Perform constrained optimization of BD^2 over the feasible parameter domain using calculus or inequalities. "
        "Identify minimal attainable BD^2 corresponding to valid rhombi, confirm it is strictly greater than degenerate value (80), and respects all constraints."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1, thinking_1_3, answer_1_3]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, optimize BD^2, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide limitations of the optimization. If correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback on optimization, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refine optimization, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    # Subtask 3: Interpret problem wording about greatest real number less than BD^2 for all rhombi
    cot_debate_instruction_2_3 = (
        "Sub-task 2.3: Interpret the problem's wording 'greatest real number less than BD^2 for all such rhombi.' "
        "Distinguish minimum, infimum, supremum, and maximum of BD^2. Analyze whether minimal BD^2 is attained or approached asymptotically, "
        "clarify implications for final answer with rigorous explanation."
    )
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_3 = self.max_round
    all_thinking_2_3 = [[] for _ in range(N_max_2_3)]
    all_answer_2_3 = [[] for _ in range(N_max_2_3)]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_debate_instruction_2_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_3):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_2, answer_2_2], cot_debate_instruction_2_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_2_3[r-1] + all_answer_2_3[r-1]
                thinking_i, answer_i = await agent(input_infos, cot_debate_instruction_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, interpret problem wording, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_3[r].append(thinking_i)
            all_answer_2_3[r].append(answer_i)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3([taskInfo] + all_thinking_2_3[-1] + all_answer_2_3[-1], "Sub-task 2.3: Provide final interpretation and answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalize interpretation, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task 2.3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
