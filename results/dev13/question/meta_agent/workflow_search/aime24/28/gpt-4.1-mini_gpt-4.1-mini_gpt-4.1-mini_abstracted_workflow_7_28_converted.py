async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    # Stage 0 - Subtask 1: Define torus T
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the torus T with major radius R=6 and minor radius r=3. "
        "Express the torus surface parametrically and implicitly, assuming the torus axis is aligned with the z-axis and the coordinate system is centered at the sphere center. "
        "Do not solve tangency conditions yet."
    )
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, defining torus T, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0.Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Stage 0 - Subtask 2: Define sphere S
    cot_instruction_0_2 = (
        "Sub-task 2: Formally define the sphere S with radius 11 centered at the origin. "
        "Express its surface equation explicitly, consistent with the torus coordinate system."
    )
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, defining sphere S, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0.Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Stage 0 - Subtask 3: Clarify configurations and vertical offset H
    debate_instruction_0_3 = (
        "Sub-task 3: Clarify and explicitly distinguish the two configurations of tangency between the torus and sphere: "
        "(a) external tangency where the torus rests outside the sphere, and "
        "(b) internal tangency where the sphere lies inside the torus hole and the torus is tangent from outside. "
        "Remove ambiguous phrasing and introduce vertical offset H between torus center and sphere center along z-axis. "
        "State assumptions about relative positioning and orientation."
    )
    debate_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0_3 = self.max_round
    all_thinking_0_3 = [[] for _ in range(N_max_0_3)]
    all_answer_0_3 = [[] for _ in range(N_max_0_3)]
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": debate_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_3):
        for i, agent in enumerate(debate_agents_0_3):
            if r == 0:
                thinking_0_3, answer_0_3 = await agent([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], debate_instruction_0_3, r, is_sub_task=True)
            else:
                input_infos_0_3 = [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2] + all_thinking_0_3[r-1] + all_answer_0_3[r-1]
                thinking_0_3, answer_0_3 = await agent(input_infos_0_3, debate_instruction_0_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, clarifying configurations, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
            all_thinking_0_3[r].append(thinking_0_3)
            all_answer_0_3[r].append(answer_0_3)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + all_thinking_0_3[-1] + all_answer_0_3[-1], "Sub-task 3: Synthesize and finalize configuration clarification." + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, clarifying configurations, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Stage 0.Subtask 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Stage 0 outputs for next stage
    stage_0_outputs = {
        "torus_def": answer_0_1.content,
        "sphere_def": answer_0_2.content,
        "config_clarification": answer_0_3.content
    }

    # Stage 1 - Subtask 1: Solve for vertical offsets H for each configuration
    cot_sc_instruction_1_1 = (
        "Sub-task 1: For each configuration (external and internal tangency), solve for the vertical offset H between the torus center and the sphere center that ensures tangency. "
        "Use geometric relations derived from tangency conditions, considering torus tube radius r=3 and sphere radius 11. "
        "Provide explicit expressions or numeric values for H in each case."
    )
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": [taskInfo, stage_0_outputs["torus_def"], stage_0_outputs["sphere_def"], stage_0_outputs["config_clarification"]],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, stage_0_outputs["torus_def"], stage_0_outputs["sphere_def"], stage_0_outputs["config_clarification"]], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, solving for vertical offsets H, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent vertical offsets H for each configuration.", is_sub_task=True)
    agents.append(f"Final Decision agent, vertical offsets H, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1.Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Stage 1 - Subtask 2: Derive and solve quadratic in z^2 for each configuration
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Using the computed vertical offsets H, derive the quadratic equation in z squared (z^2) from the tangency condition for each configuration. "
        "Solve the quadratic completely, find all real roots, and verify domain validity (z^2 >= 0 and within torus tube bounds)."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": [taskInfo, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, answer_1_1.content], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, solving quadratic in z^2, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent roots for z^2.", is_sub_task=True)
    agents.append(f"Final Decision agent, roots of quadratic, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1.Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Stage 1 - Subtask 3: Calculate tangent circle radii r_i and r_o
    cot_sc_instruction_1_3 = (
        "Sub-task 3: For each valid root z obtained, calculate the radius of the tangent circle on the torus surface (r_i or r_o) using the torus parametric form and vertical offset H. "
        "Clearly distinguish which radius corresponds to which configuration."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": [taskInfo, answer_1_1.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, answer_1_1.content, answer_1_2.content], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, calculating tangent circle radii, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Sub-task 3: Synthesize and finalize tangent circle radii.", is_sub_task=True)
    agents.append(f"Final Decision agent, tangent circle radii, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Stage 1.Subtask 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # Stage 1 - Subtask 4: Verify geometric consistency
    debate_instruction_1_4 = (
        "Sub-task 4: Verify the geometric consistency of the computed tangent circles and their radii by checking that the torus and sphere surfaces are tangent along these circles. "
        "Confirm that vertical offsets, roots, and radii satisfy all problem constraints and assumptions."
    )
    debate_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_4 = self.max_round
    all_thinking_1_4 = [[] for _ in range(N_max_1_4)]
    all_answer_1_4 = [[] for _ in range(N_max_1_4)]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": debate_instruction_1_4,
        "context": [taskInfo, answer_1_1.content, answer_1_2.content, answer_1_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_4):
        for i, agent in enumerate(debate_agents_1_4):
            if r == 0:
                thinking_1_4, answer_1_4 = await agent([taskInfo, answer_1_1.content, answer_1_2.content, answer_1_3.content], debate_instruction_1_4, r, is_sub_task=True)
            else:
                input_infos_1_4 = [taskInfo, answer_1_1.content, answer_1_2.content, answer_1_3.content] + all_thinking_1_4[r-1] + all_answer_1_4[r-1]
                thinking_1_4, answer_1_4 = await agent(input_infos_1_4, debate_instruction_1_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying geometric consistency, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
            all_thinking_1_4[r].append(thinking_1_4)
            all_answer_1_4[r].append(answer_1_4)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + all_thinking_1_4[-1] + all_answer_1_4[-1], "Sub-task 4: Synthesize and finalize verification of geometric consistency." + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, geometric verification, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Stage 1.Subtask 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)

    # Stage 2 - Subtask 1: Compute difference r_i - r_o as reduced fraction
    cot_instruction_2_1 = (
        "Sub-task 1: Compute the difference r_i - r_o using the expressions obtained for the two tangent circle radii. "
        "Simplify the result to a reduced fraction m/n where m and n are relatively prime positive integers."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": [taskInfo, answer_1_3.content, answer_1_4.content],
        "agent_collaboration": "SC_CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, answer_1_3.content, answer_1_4.content], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing difference r_i - r_o, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2.Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Stage 2 - Subtask 2: Calculate m + n from simplified fraction
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Calculate the sum m + n from the simplified fraction m/n obtained in the previous subtask. "
        "Verify correctness of arithmetic and simplification steps to ensure final answer accuracy. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": [taskInfo, thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, calculating m+n, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback_2_2.content}; correct: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining m+n, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2.Subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
