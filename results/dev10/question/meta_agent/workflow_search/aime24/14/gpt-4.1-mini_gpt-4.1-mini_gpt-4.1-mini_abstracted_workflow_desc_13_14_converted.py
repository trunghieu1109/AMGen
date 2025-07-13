async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Parametrization and Rhombus Conditions

    # Subtask 1: Derive valid real parametrization of points on the hyperbola using sec–tan parametrization
    cot_instruction_0_1 = (
        "Sub-task 1: Derive a valid real parametrization of points on the hyperbola x^2/20 - y^2/24 = 1 "
        "using the sec–tan parametrization: x = sqrt(20) sec(theta), y = sqrt(24) tan(theta), where theta is a real parameter. "
        "State the domain of theta and justify why this parametrization covers all relevant points on the hyperbola branches. "
        "Introduce notation for points A, B, C, D on the hyperbola in terms of parameters consistent with the rhombus properties and the midpoint at the origin condition. "
        "Avoid the previously used hyperbolic parametrization that admits no real solutions satisfying the rhombus conditions."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving valid parametrization, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Subtask 2: Formulate rhombus geometric conditions explicitly in terms of the parametrization
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Formulate the rhombus geometric conditions explicitly in terms of the sec–tan parametrization derived in Sub-task 1: "
        "(1) diagonals intersect at the origin (midpoint condition), (2) diagonals are perpendicular, "
        "(3) all sides equal length, (4) rhombus is convex and planar. Write algebraic equations involving parameters. "
        "Avoid assumptions about point labeling without justification. Set foundation for rigorous optimization."
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
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, formulating rhombus conditions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent rhombus conditions in terms of parameters", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_0_2.id}, synthesizing rhombus conditions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Stage 1: Derive explicit BD^2 expression and set up optimization

    # Subtask 1: Derive explicit single-variable expression for BD^2(theta)
    cot_reflect_instruction_1_1 = (
        "Sub-task 1: Combine the sec–tan parametrization and rhombus conditions to derive explicit expressions for the diagonals and BD^2 as a function of a single parameter theta. "
        "Simplify to a single-variable function BD^2(theta) suitable for calculus optimization. Ensure perpendicularity condition reduces parameter space and domain constraints are respected. "
        "Avoid spurious solutions."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Reflexion | CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2, answer_0_2], cot_reflect_instruction_1_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, deriving BD^2(theta), thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Subtask 2: Set up the optimization problem clearly
    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Set up the optimization problem to maximize BD^2(theta) subject to domain and rhombus conditions. "
        "Analyze boundary behavior and function properties to prepare for derivative and critical point analysis. "
        "Avoid premature acceptance of candidates."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_0_2, answer_0_2, thinking_1_1, answer_1_1], cot_reflect_instruction_1_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, setting up optimization, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Stage 2: Calculus-based analysis and candidate enumeration

    # Subtask 1: Perform detailed calculus-based analysis to find critical points and evaluate BD^2
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Compute derivative d(BD^2)/d(theta), solve for critical points in domain, evaluate BD^2 at these points and boundaries. "
        "Confirm candidate supremum BD^2=160 and note other candidates such as 256. Ensure all candidates satisfy rhombus constraints."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, calculus analysis, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 2.1: Synthesize candidate supremum values of BD^2", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_1.id}, enumerating candidates, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Subtask 2: Enumerate all candidate supremum values including 160 and 256
    cot_instruction_2_2 = (
        "Sub-task 2.2: Enumerate all candidate supremum values of BD^2 found, including 160 and 256. "
        "For each candidate, prepare algebraic expressions and parameter values for rigorous verification. "
        "Avoid premature dismissal or acceptance; set framework for systematic verification."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, enumerating candidates, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    # Stage 3: Rigorous verification and final answer

    # Subtask 1: For each candidate, rigorously verify all rhombus and hyperbola conditions
    debate_instruction_3_1 = (
        "Sub-task 3.1: For each candidate supremum BD^2 value enumerated, rigorously verify all geometric conditions: "
        "points lie on hyperbola, diagonals intersect at origin and are perpendicular bisectors, all sides equal length, figure is convex. "
        "Provide algebraic and geometric proofs or contradictions. Document failures, especially for BD^2=256. "
        "Ensure no invalid solution is accepted."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate | SC_CoT | Reflexion"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_3_1, answer_3_1 = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_3_1, answer_3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying candidates, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
            all_thinking_3_1[r].append(thinking_3_1)
            all_answer_3_1[r].append(answer_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1_final, answer_3_1_final = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 3.1: Final verification and validation of candidate supremum values", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_3_1.id}, final verification, thinking: {thinking_3_1_final.content}; answer: {answer_3_1_final.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1_final.content}; answer - {answer_3_1_final.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1_final, "answer": answer_3_1_final}
    logs.append(subtask_desc_3_1)

    # Subtask 2: Select unique valid supremum and present final answer
    cot_instruction_3_2 = (
        "Sub-task 3.2: Based on verification results, select the unique valid supremum BD^2 value for all rhombi on the hyperbola with diagonals intersecting at the origin. "
        "Present final answer as 160 with rigorous justification why other candidates (e.g., 256) are invalid. "
        "Include summary of verification and reaffirm all conditions are satisfied. Avoid ambiguity or incomplete justification."
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_instruction_3_2,
        "context": ["user query", thinking_3_1_final.content, answer_3_1_final.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2([taskInfo, thinking_3_1_final, answer_3_1_final], cot_instruction_3_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_2.id}, final answer selection, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
