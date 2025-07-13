async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Define hyperbola and rhombus properties, combine constraints

    # Sub-task 1: Define hyperbola domain (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the hyperbola and express the domain of points A, B, C, and D on it. "
        "State the hyperbola equation x^2/20 - y^2/24 = 1 and clarify that each point must satisfy this equation. "
        "Discuss the hyperbola's symmetry and center at the origin without solving or simplifying."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining hyperbola domain, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Express rhombus properties algebraically (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Formally express the rhombus properties relevant to points A, B, C, and D: equal side lengths, diagonals bisecting each other at the origin, and diagonals being perpendicular. "
        "Represent these conditions algebraically in terms of the coordinates of the points. Avoid mixing these with the hyperbola constraints at this stage."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, expressing rhombus properties, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Sub-task 3: Combine hyperbola and rhombus constraints (SC_CoT)
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Combine the hyperbola constraints and rhombus properties to derive a simplified system of equations relating the coordinates of points A, B, C, and D. "
        "Use symmetry about the origin and midpoint condition to reduce variables. Explicitly state assumptions about point labeling and ordering to avoid ambiguity. "
        "Avoid attempting optimization here; focus on obtaining a consistent algebraic system."
    )
    N_sc_0_3 = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_3)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_3):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_1, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, combining constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent algebraic system for the combined constraints.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_0_3.id}, synthesizing combined constraints, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Derive BD^2 expression and perpendicularity condition

    # Sub-task 1: Expression for BD^2 (CoT)
    cot_instruction_1_1 = (
        "Sub-task 1: Derive an explicit expression for the squared length of diagonal BD, BD^2, in terms of the parameters or variables identified in Stage 0. "
        "Ensure this expression incorporates the constraints from the hyperbola and rhombus conditions. Avoid solving for extrema here."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, deriving BD^2 expression, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Sub-task 2: Incorporate perpendicularity condition (SC_CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Derive the perpendicularity condition of the diagonals in terms of the chosen parameters and incorporate it into the expression for BD^2. "
        "Express all constraints in a form suitable for algebraic or calculus-based optimization. Avoid solving optimization here."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_3.content, thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_0_3, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, incorporating perpendicularity, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent expression including perpendicularity.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_2.id}, synthesizing perpendicularity condition, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 2: Root validation, supremum identification, geometric validity, and final summary

    # Sub-task 1: Root validation and domain filtering (Debate)
    debate_instr_2_1 = (
        "Sub-task 1: Solve the quadratic equation 11 t^2 - 1440 t + 14400 = 0 exactly, enumerate all real roots. "
        "Apply domain constraints (A^2 >= 20, B^2 >= 20) to discard invalid roots. Substitute surviving roots back into the original hyperbola and perpendicularity equations to verify all constraints. "
        "Confirm the corresponding rhombus is nondegenerate and convex. This subtask is dedicated solely to root validation and domain filtering. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_2_1,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2], debate_instr_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2] + all_thinking_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instr_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, root validation, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1], "Sub-task 1: Final decision on root validation and domain filtering. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_1.id}, finalizing root validation, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Sub-task 2: Supremum identification and attainability analysis (Reflexion)
    reflect_inst_2_2 = (
        "Sub-task 2: Identify the supremum (greatest lower bound) of BD^2 under the combined constraints based on validated roots. "
        "Provide detailed algebraic and geometric proofs that this supremum is not attained by any rhombus on the hyperbola but can be approached arbitrarily closely. "
        "Analyze boundary behavior and parameter limits to distinguish between supremum and maximum values. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflect_inst_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_inst_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, analyzing supremum and attainability, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback on supremum analysis, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_inst_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining supremum analysis, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Sub-task 3: Geometric validity near supremum (SC_CoT)
    cot_sc_instruction_2_3 = (
        "Sub-task 3: Analyze the geometric validity of rhombi near the supremum value of BD^2. "
        "Verify convexity, nondegeneracy, and the impact of point ordering and labeling on the solution. "
        "Confirm that all geometric and algebraic constraints hold in the limiting cases approaching the supremum. Avoid re-solving the optimization; focus on geometric interpretation and validation."
    )
    N_sc_2_3 = self.max_sc
    cot_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_3)]
    possible_answers_2_3 = []
    possible_thinkings_2_3 = []
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_sc_instruction_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_3):
        thinking_i, answer_i = await cot_agents_2_3[i]([taskInfo, thinking_2_2], cot_sc_instruction_2_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_3[i].id}, analyzing geometric validity near supremum, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_3.append(answer_i)
        possible_thinkings_2_3.append(thinking_i)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3([taskInfo] + possible_thinkings_2_3, "Sub-task 3: Synthesize geometric validity near supremum.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_3.id}, synthesizing geometric validity, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task 2.3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    # Sub-task 4: Final summary (SC_CoT)
    cot_sc_instruction_2_4 = (
        "Sub-task 4: Summarize the findings explicitly: state the supremum value of BD^2, clarify that this supremum is not attained by any rhombus on the hyperbola, "
        "and conclude that the greatest real number less than BD^2 for all such rhombi can be arbitrarily close to but less than the supremum. "
        "Include a clear explanation of the difference between supremum and maximum in this context."
    )
    N_sc_2_4 = self.max_sc
    cot_agents_2_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_4)]
    possible_answers_2_4 = []
    possible_thinkings_2_4 = []
    subtask_desc_2_4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_sc_instruction_2_4,
        "context": ["user query", thinking_2_2.content, thinking_2_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_4):
        thinking_i, answer_i = await cot_agents_2_4[i]([taskInfo, thinking_2_2, thinking_2_3], cot_sc_instruction_2_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_4[i].id}, summarizing final findings, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_4.append(answer_i)
        possible_thinkings_2_4.append(thinking_i)
    final_decision_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_4, answer_2_4 = await final_decision_agent_2_4([taskInfo] + possible_thinkings_2_4, "Sub-task 4: Final summary of supremum and greatest real number less than BD^2.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_4.id}, finalizing summary, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
    sub_tasks.append(f"Sub-task 2.4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking_2_4, "answer": answer_2_4}
    logs.append(subtask_desc_2_4)
    print("Step 2.4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_4, answer_2_4, sub_tasks, agents)
    return final_answer, logs
