async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify and verify the geometric parameters of the torus and sphere, "
        "including major radius R=6, minor radius r=3 of the torus, and radius 11 of the sphere. "
        "Establish a clear coordinate system and define what it means for the torus to be externally tangent to the sphere along a circle. "
        "Clarify the two distinct configurations of the torus resting on the outside of the sphere and the geometric meaning of the tangent circles with radii r_i and r_o. "
        "Avoid ambiguity and ensure consistent notation and setup for subsequent algebraic work."
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
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Formulate the exact geometric conditions and constraints for external tangency between the torus and sphere along a circle in each configuration. "
        "Derive the system of equations relating the vertical displacement c of the torus center, major radius R=6, minor radius r=3, and sphere radius 11. "
        "Explicitly set up the two key equations: (sqrt(x^2 + y^2) - R)^2 + c^2 = r^2 (torus generating circle) and x^2 + y^2 + c^2 = 11^2 (sphere surface). "
        "Explain how these relate to the tangent circle radii r_i and r_o. Emphasize correct interpretation of addition or subtraction of r in distance conditions."
    )
    N_sc = self.max_sc
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    possible_thinkings_2 = []
    for i in range(N_sc):
        thinking2, answer2 = await cot_sc_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, formulating geometric conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 2: Synthesize and choose the most consistent geometric conditions and equations." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3a = (
        "Sub-task 3a: Solve the vertical displacement equations for the torus center c in each configuration by handling the two cases corresponding to the torus resting on the sphere in two distinct ways. "
        "Explicitly solve the equations sqrt((R ± r)^2 + c^2) = 11 ∓ r for c_i and c_o, showing all algebraic steps and verifying validity of solutions. "
        "Document reasoning for choosing correct roots."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round_3a = self.max_round
    all_thinking_3a = [[] for _ in range(N_round_3a)]
    all_answer_3a = [[] for _ in range(N_round_3a)]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": debate_instr_3a,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round_3a):
        for i, agent in enumerate(debate_agents_3a):
            if r == 0:
                thinking3a, answer3a = await agent([taskInfo, thinking2, answer2], debate_instr_3a, r, is_sub_task=True)
            else:
                input_infos_3a = [taskInfo, thinking2, answer2] + all_thinking_3a[r-1] + all_answer_3a[r-1]
                thinking3a, answer3a = await agent(input_infos_3a, debate_instr_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving vertical displacements, thinking: {thinking3a.content}; answer: {answer3a.content}")
            all_thinking_3a[r].append(thinking3a)
            all_answer_3a[r].append(answer3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo, thinking2, answer2] + all_thinking_3a[-1] + all_answer_3a[-1], "Sub-task 3a: Finalize vertical displacement solutions." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    debate_instr_3b = (
        "Sub-task 3b: Using vertical displacements c_i and c_o from subtask 3a, compute the radii r_i and r_o of the tangent circles formed by the external tangency of the torus and sphere. "
        "Substitute c_i and c_o into the sphere equation x^2 + y^2 = 11^2 - c^2 to find the radius of the tangent circle on the sphere, which equals the radius of the tangent circle on the torus. "
        "Explicitly distinguish these tangent circle radii from the torus generating circle radius r and verify correctness. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round_3b = self.max_round
    all_thinking_3b = [[] for _ in range(N_round_3b)]
    all_answer_3b = [[] for _ in range(N_round_3b)]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": debate_instr_3b,
        "context": ["user query", thinking3a, answer3a],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking3b, answer3b = await agent([taskInfo, thinking3a, answer3a], debate_instr_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking3a, answer3a] + all_thinking_3b[r-1] + all_answer_3b[r-1]
                thinking3b, answer3b = await agent(input_infos_3b, debate_instr_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing tangent circle radii, thinking: {thinking3b.content}; answer: {answer3b.content}")
            all_thinking_3b[r].append(thinking3b)
            all_answer_3b[r].append(answer3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3b, answer3b = await final_decision_agent_3b([taskInfo, thinking3a, answer3a] + all_thinking_3b[-1] + all_answer_3b[-1], "Sub-task 3b: Finalize tangent circle radii calculations." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Calculate the difference r_i - r_o using the derived values from subtask 3b. "
        "Express this difference as a reduced fraction m/n where m and n are relatively prime positive integers. "
        "Compute and return the sum m + n. Include explicit fraction simplification steps. "
        "Rely solely on rigorously verified inputs to avoid propagating errors."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3b, answer3b],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3b, answer3b], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating difference and fraction simplification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
