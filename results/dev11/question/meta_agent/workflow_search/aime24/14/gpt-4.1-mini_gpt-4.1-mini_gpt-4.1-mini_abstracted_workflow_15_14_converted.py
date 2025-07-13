async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # --------------------------------------------------------------------------------------------------------------
    # Stage 0: Formal Representation and Parameterization
    # Subtask 1: Formal representation of hyperbola and rhombus conditions
    cot_instruction_0_1 = (
        "Subtask 0_1: Formally represent the hyperbola equation and rhombus conditions. "
        "Define coordinates for points A, B, C, D on the hyperbola x^2/20 - y^2/24 = 1, "
        "with diagonals intersecting at the origin. Use vector notation to express that origin is midpoint of AC and BD, "
        "implying A = -C and B = -D. State explicitly the perpendicularity condition of diagonals and equality of sides. "
        "Emphasize symmetry and geometric meaning of constraints."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage0_subtask1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formal representation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Subtask stage0_subtask1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    # Subtask 2: Derive algebraic relationships and parameterize points on hyperbola
    cot_sc_instruction_0_2 = (
        "Subtask 0_2: Based on stage0_subtask1, derive algebraic relationships from midpoint and perpendicularity conditions. "
        "Express equality of side lengths in terms of coordinates of A and B (with C = -A, D = -B). "
        "Introduce parameterization of points on hyperbola using hyperbolic functions: x = sqrt(20)*cosh t, y = sqrt(24)*sinh t. "
        "Derive expressions for vectors representing sides and diagonals with detailed algebraic steps. "
        "Prepare expressions for BD^2 derivation. Include intermediate algebraic verification."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage0_subtask2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, algebraic relationships, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Subtask 0_2: Synthesize algebraic relationships and parameterization", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_0_2.id}, algebraic synthesis, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Subtask stage0_subtask2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0_2: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 1: BD^2 Derivation and Constraint Analysis
    # Subtask 1: Derive explicit expression for BD^2 in terms of hyperbolic parameters
    cot_sc_instruction_1_1 = (
        "Subtask 1_1: Derive step-by-step explicit expression for BD^2 solely in terms of hyperbolic parameters t1, t2 representing points B and D on the hyperbola. "
        "Use parameterization and orthogonality of diagonals to eliminate redundant variables. Apply hyperbolic identities explicitly. "
        "Include algebraic sanity check by substituting t2=0 to verify BD^2=80. Confirm final formula BD^2=80 + 16 sinh^2(t2)."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage1_subtask1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, BD^2 derivation, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Subtask 1_1: Synthesize BD^2 explicit formula", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_1.id}, BD^2 formula synthesis, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Subtask stage1_subtask1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    # Subtask 2: Derive constraint on t2 from rhombus condition and hyperbola, compute minimal BD^2
    cot_sc_instruction_1_2 = (
        "Subtask 1_2: Derive constraint on parameter t2 from rhombus condition and hyperbola equation. "
        "Express tanh t2 = ±5/6, rewrite sinh^2 t2 in terms of tanh^2 t2 using hyperbolic identities. "
        "Compute minimal BD^2 by substituting constraint into BD^2 formula from stage1_subtask1. "
        "Perform numeric evaluation with detailed steps and verify consistency with geometric constraints."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1, thinking_0_2, answer_0_2], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, parameter constraint and minimal BD^2, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Subtask 1_2: Synthesize parameter constraint and minimal BD^2", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_2.id}, minimal BD^2 computation, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Subtask stage1_subtask2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1_2: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 2: Feasibility Analysis and Final Numeric Result
    # Subtask 1: Analyze feasible domain and confirm minimal BD^2 corresponds to infimum
    debate_instruction_2_1 = (
        "Subtask 2_1: Analyze feasible domain of parameters and confirm that minimal BD^2 from stage1_subtask2 corresponds to the infimum of BD^2 over all rhombi inscribed on the hyperbola with diagonals intersecting at origin. "
        "Use calculus or algebraic reasoning to exclude degenerate or non-convex rhombi. Verify rhombus and hyperbola conditions hold for extremal configuration. "
        "Clarify problem interpretation emphasizing infimum (greatest real number less than all BD^2)."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage2_subtask1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, feasibility analysis, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Subtask 2_1: Final feasibility and infimum confirmation", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_1.id}, feasibility confirmation, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Subtask stage2_subtask1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    # Subtask 2: Provide final numeric or closed-form expression for greatest real number less than BD^2
    debate_instruction_2_2 = (
        "Subtask 2_2: Provide final numeric or closed-form expression for greatest real number less than BD^2 for all such rhombi, based on verified minimal value from stage2_subtask1. "
        "Simplify numeric result with exact arithmetic or well-justified approximations. Include final verification confirming no contradictions or overlooked cases. "
        "Summarize geometric interpretation relating to hyperbola and rhombus configuration. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_2 = self.max_round
    all_thinking_2_2 = [[] for _ in range(N_max_2_2)]
    all_answer_2_2 = [[] for _ in range(N_max_2_2)]
    subtask_desc_2_2 = {
        "subtask_id": "stage2_subtask2",
        "instruction": debate_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_2):
        for i, agent in enumerate(debate_agents_2_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_1, answer_2_1], debate_instruction_2_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_1, answer_2_1] + all_thinking_2_2[r-1] + all_answer_2_2[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, final numeric result, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_2[r].append(thinking_i)
            all_answer_2_2[r].append(answer_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + all_thinking_2_2[-1] + all_answer_2_2[-1], "Subtask 2_2: Final numeric answer and geometric interpretation", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_2.id}, final numeric answer, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Subtask stage2_subtask2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2_2: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 3: Final Verification
    # Subtask 1: Comprehensive verification of algebraic identities, numeric computations, and geometric reasoning
    cot_reflect_instruction_3_1 = (
        "Subtask 3_1: Perform comprehensive final verification of all critical algebraic identities, numeric computations, and geometric reasoning from previous subtasks. "
        "Cross-check BD^2 derivation, parameter constraints, and optimization results by plugging in test values (e.g., t2=0, t2->infinity). "
        "Confirm all assumptions and simplifications hold. Provide detailed report confirming correctness or identifying inconsistencies with suggestions. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_1 = self.max_round
    cot_inputs_3_1 = [taskInfo, thinking_2_2, answer_2_2, thinking_1_2, answer_1_2, thinking_1_1, answer_1_1, thinking_0_2, answer_0_2]
    subtask_desc_3_1 = {
        "subtask_id": "stage3_subtask1",
        "instruction": cot_reflect_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content, thinking_1_2.content, answer_1_2.content, thinking_1_1.content, answer_1_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, final verification, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    for i in range(N_max_3_1):
        feedback_3_1, correct_3_1 = await critic_agent_3_1([taskInfo, thinking_3_1, answer_3_1], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1.id}, feedback: {feedback_3_1.content}; correct: {correct_3_1.content}")
        if correct_3_1.content == "True":
            break
        cot_inputs_3_1.extend([thinking_3_1, answer_3_1, feedback_3_1])
        thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining verification, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Subtask stage3_subtask1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3_1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
