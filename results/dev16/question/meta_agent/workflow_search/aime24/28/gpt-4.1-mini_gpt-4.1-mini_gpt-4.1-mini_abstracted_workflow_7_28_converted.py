async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Define surfaces and derive tangency condition

    # Subtask 1: Define torus and sphere equations with assumptions
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the torus T with major radius R=6 and minor radius r=3, "
        "and the sphere S with radius 11, by deriving their implicit equations in a fixed 3D coordinate system. "
        "Assume the torus axis aligns with the z-axis and the sphere center is at the origin. "
        "Do not assume tangency or intersection yet. Provide explicit equations and clear assumptions."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining torus and sphere, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Derive explicit tangency condition (intersection + parallel gradients)
    cot_instruction_0_2 = (
        "Sub-task 2: Using the torus and sphere equations from Sub-task 1, derive the explicit condition for external tangency between T and S along a circle. "
        "Formulate the system of equations representing their intersection and the condition that their gradients (normals) are parallel at points of tangency. "
        "Simplify this system to a single transcendental equation in terms of a parameter (e.g., cos(phi)) characterizing the circle of tangency. "
        "Provide detailed algebraic steps without skipping or assuming trivial solutions."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, deriving tangency condition, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Subtask 3: Identify two distinct tangency configurations and relate to radii r_i and r_o
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Based on the transcendental tangency equation from Sub-task 2, identify and clearly distinguish the two distinct external tangency configurations of the torus resting on the sphere. "
        "Define the geometric parameters (e.g., angles phi_i and phi_o) characterizing each tangency circle and relate these parameters to the radii r_i and r_o on the torus surface. "
        "Avoid assuming these radii without explicit derivation."
    )
    N_sc_0_3 = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_3)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_3):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, identifying tangency configurations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent and correct identification of tangency configurations and radii relations.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Subtask 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    agents.append(f"Final Decision agent, synthesizing tangency configurations, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Solve tangency equation for phi_i and phi_o, compute r_i and r_o

    # Subtask 1: Solve for phi_i and compute r_i
    cot_instruction_1_1 = (
        "Sub-task 1: Solve the transcendental tangency equation derived in Stage 0 to find the first solution phi_i corresponding to one external tangency configuration. "
        "Provide detailed algebraic or numeric reasoning and justification. Then compute the radius r_i = R + r * cos(phi_i) of the circle of tangency on the torus. "
        "Ensure consistency with the geometric setup and previous results."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, solving for phi_i and computing r_i, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Solve for phi_o and compute r_o
    cot_instruction_1_2 = (
        "Sub-task 2: Solve the transcendental tangency equation to find the second solution phi_o corresponding to the other external tangency configuration. "
        "Provide detailed algebraic or numeric reasoning and justification. Then compute the radius r_o = R + r * cos(phi_o) of the circle of tangency on the torus. "
        "Verify that this solution is distinct and consistent with problem constraints."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_0_3, answer_0_3], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, solving for phi_o and computing r_o, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Subtask 3: Verification and cross-check via Debate
    debate_instr_1_3 = (
        "Sub-task 3: Independently verify and cross-check the solutions phi_i, phi_o and corresponding radii r_i, r_o obtained in Subtasks 1 and 2. "
        "Use multiple agents to solve the tangency equations separately and a moderator agent to compare results for consistency and correctness. "
        "Identify and resolve any discrepancies before proceeding."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_3 = self.max_round
    all_thinking_1_3 = [[] for _ in range(N_max_1_3)]
    all_answer_1_3 = [[] for _ in range(N_max_1_3)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instr_1_3,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_3):
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking_1_3, answer_1_3 = await agent([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], debate_instr_1_3, r, is_sub_task=True)
            else:
                input_infos_1_3 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2] + all_thinking_1_3[r-1]
                thinking_1_3, answer_1_3 = await agent(input_infos_1_3, debate_instr_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying solutions, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
            all_thinking_1_3[r].append(thinking_1_3)
            all_answer_1_3[r].append(answer_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3_final, answer_1_3_final = await final_decision_agent_1_3([taskInfo] + all_thinking_1_3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final verified answer for phi_i, phi_o, r_i, and r_o.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 3 output: thinking - {thinking_1_3_final.content}; answer - {answer_1_3_final.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3_final, "answer": answer_1_3_final}
    logs.append(subtask_desc_1_3)
    agents.append(f"Final Decision agent, verifying solutions, thinking: {thinking_1_3_final.content}; answer: {answer_1_3_final.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2: Compute difference and final sum

    # Subtask 1: Compute difference r_i - r_o and reduce fraction
    cot_instruction_2_1 = (
        "Sub-task 1: Using the verified values of r_i and r_o from Stage 1, compute the difference r_i - r_o. "
        "Simplify this difference to a reduced fraction m/n where m and n are relatively prime positive integers. "
        "Provide a clear, step-by-step simplification process including any necessary rationalization or reduction."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3_final.content, answer_1_3_final.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_3_final, answer_1_3_final], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing difference and reducing fraction, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Subtask 2: Calculate sum m + n and confirm final answer with Reflexion
    reflect_inst_2_2 = (
        "Sub-task 2: Calculate the sum m + n from the simplified fraction obtained in Sub-task 1. "
        "Confirm that the fraction is in lowest terms before summation. Provide a concise final statement of the answer. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflect_inst_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_inst_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, calculating sum m+n and confirming answer, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_inst_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining answer, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 Subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
