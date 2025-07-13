async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify and confirm the fundamental parameters and geometric setup from the problem statement. "
        "Determine the torus parameters — major radius R = 6 and minor radius r = 3 — and the sphere radius = 11. "
        "Establish a clear coordinate system and orientation assumptions: place the sphere centered at the origin, with the torus axis vertical along the z-axis. "
        "Clarify the torus generation by revolving the circle in the xy-plane around the z-axis. "
        "Avoid assuming tangency points or configurations prematurely. This subtask sets the foundational geometric framework and notation for all subsequent analysis."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing geometric setup, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = (
        "Sub-task 2: Formulate the precise geometric conditions for external tangency between the torus and the sphere in the two distinct configurations. "
        "2a) Express the torus parametrization explicitly as a function of parameters (theta, phi), and write the squared distance from the sphere center to points on the torus surface as a function of these parameters and an unknown vertical offset delta_z of the torus relative to the sphere. "
        "2b) Impose the tangency condition by requiring that the sphere radius equals the distance from its center to the torus surface along a circle, and that the tangent planes of the torus and sphere align at these points. "
        "Derive the implicit equations relating theta (the angle parameterizing the torus generating circle), delta_z, and the sphere radius. Avoid premature assumptions or substitutions such as R ± r for the tangent circle radii."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, formulating tangency conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent and correct implicit equations for tangency conditions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)

    cot_instruction_3 = (
        "Sub-task 3: Solve the implicit system of equations derived in Sub-task 2 to find the two distinct solutions for theta (denote them theta_i and theta_o) and the corresponding vertical offsets delta_z that satisfy the tangency conditions. "
        "Use symbolic manipulation or numerical approximation methods to solve the nonlinear trigonometric equations accurately. "
        "Ensure both solutions correspond to valid geometric configurations where the torus is externally tangent to the sphere along a circle. "
        "Provide explicit values or expressions for theta_i, theta_o, and delta_z. Avoid accepting approximate or intuitive guesses without verification."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2.content, answer2.content], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, solving implicit equations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)

    cot_instruction_4 = (
        "Sub-task 4: Using the solutions (theta_i, theta_o) and delta_z from Sub-task 3, compute the exact radii r_i and r_o of the tangent circles on the torus surface. "
        "Recall that the radius of a circle on the torus surface corresponding to parameter theta is r * |sin(theta)|. "
        "Evaluate these expressions precisely and simplify to reduced fractional form where possible. "
        "Confirm that these radii satisfy the tangency conditions and are consistent with the sphere radius and torus parameters. "
        "Avoid substituting R ± r directly or prematurely simplifying before solving the implicit equations."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3.content, answer3.content], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, computing tangent circle radii, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)

    debate_instruction_5 = (
        "Sub-task 5: Calculate the difference r_i - r_o using the exact values obtained in Sub-task 4. "
        "Simplify this difference to a reduced fraction m/n, ensuring that m and n are relatively prime positive integers. "
        "Verify the fraction rigorously by cross-checking with the geometric constraints and prior calculations. "
        "Avoid sign errors, incorrect simplifications, or accepting intuitive but incorrect numeric results. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4.content, answer4.content], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4.content, answer4.content] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating difference fraction, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Synthesize and finalize the reduced fraction for the difference r_i - r_o.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)

    debate_instruction_6 = (
        "Sub-task 6: Compute the sum m + n from the reduced fraction m/n found in Sub-task 5. "
        "Provide the final answer clearly. Perform a verification step to confirm that this sum is consistent with the problem’s conditions, the torus and sphere parameters, and the tangency configurations. "
        "Include a brief reflexion discussing the reasonableness of the result, such as checking limiting cases or geometric intuition, to ensure robustness of the solution. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5.content, answer5.content], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5.content, answer5.content] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing final sum, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Finalize and verify the sum m + n.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs