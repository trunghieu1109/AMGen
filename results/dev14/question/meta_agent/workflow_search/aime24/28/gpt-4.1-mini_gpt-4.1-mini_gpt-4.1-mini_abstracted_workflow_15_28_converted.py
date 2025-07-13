async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally define the torus T and sphere S in a 3D coordinate system. "
        "Specify the torus parameters: major radius R = 6, minor radius r = 3, and the sphere radius 11. "
        "Establish clear orientation and position assumptions, including aligning the torus axis of revolution with the z-axis and placing the sphere center at the origin. "
        "Avoid any assumptions about the tangent circle radii at this stage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining torus and sphere, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Express the geometric conditions for external tangency between the torus and the sphere. "
        "Formulate the relationship between the center of the torus generating circle (at coordinates (6, 0, h)) and the sphere center (0,0,0), incorporating the sphere radius and the torus minor radius. "
        "Clearly state the distance constraints that must hold for tangency along a circle, avoiding any premature numeric substitutions or assumptions about h."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formulating tangency conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3a = (
        "New Sub-task 3a: Set up the coordinate-geometry equations for the cross-section. "
        "Define the torus generating circle center at (6, 0, h). Impose the distance condition from the sphere center: sqrt(6^2 + h^2) = 11 ± 3. "
        "Solve for the two possible values of h representing the two tangent configurations."
    )
    N = self.max_sc
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3a = []
    possible_thinkings_3a = []
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, solving for h values, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a)
        possible_thinkings_3a.append(thinking3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo] + possible_thinkings_3a, "Sub-task 3a: Synthesize and choose the most consistent and correct solutions for h values." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_instruction_3b = (
        "New Sub-task 3b: For each value of h found in Sub-task 3a, compute the radius of the circle of tangency on the sphere using the formula r = sqrt(11^2 - h^2). "
        "Clearly explain the geometric meaning of this radius and how it relates to the circle of tangency on the torus surface. "
        "Avoid mixing up the sphere radius and torus parameters."
    )
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", thinking3a.content],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, computing tangent circle radii on sphere, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Translate the computed tangent circle radii on the sphere back into the corresponding radii r_i and r_o of the tangent circles on the torus surface. "
        "Use the torus parameters and the vertical offsets h to derive explicit formulas for r_i and r_o, ensuring the sphere radius is incorporated correctly. "
        "Avoid assuming r_i = R + r or r_o = R - r without justification."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3b.content, thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3b, thinking1], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, translating sphere radii to torus radii, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent and correct formulas for r_i and r_o." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Symbolically compute the difference r_i - r_o using the formulas derived in Sub-task 4. "
        "Simplify the expression as much as possible to a rational fraction m/n, where m and n are positive integers with no common factors. "
        "Avoid numeric approximations and ensure the fraction is in lowest terms."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, computing difference r_i - r_o, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_thinkings_5, "Sub-task 5: Synthesize and choose the simplest rational fraction for r_i - r_o." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    debate_instr_6 = (
        "Sub-task 6: Verify the correctness and geometric consistency of the computed radii r_i and r_o and their difference. "
        "Check that the tangent circle radii satisfy the sphere and torus surface equations and that the tangency conditions (including matching normals) hold. "
        "Avoid superficial checks such as only gcd simplification; instead, perform rigorous geometric validation. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instr_6,
        "context": ["user query", thinking5.content, thinking4.content, thinking1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, thinking4, thinking1], debate_instr_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, thinking4, thinking1] + all_thinking6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instr_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying geometric consistency, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1], "Sub-task 6: Final verification and geometric consistency check." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    reflect_inst_7 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_7 = (
        "Sub-task 7: Perform a reflexion step to critically evaluate the entire solution process and final results. "
        "Identify any potential oversights or errors in assumptions, derivations, or simplifications. "
        "If fundamental errors are detected, require re-derivation or correction of the affected subtasks. Otherwise, confirm the validity of the final fraction m/n. "
        + reflect_inst_7
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6.content, answer6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, critically evaluating solution, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        critic_inst_7 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7], critic_inst_7, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining solution, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_sc_instruction_8 = (
        "Sub-task 8: Calculate the sum m + n of the numerator and denominator of the simplified fraction representing r_i - r_o. "
        "Present the final answer clearly and concisely, ensuring it is consistent with all prior verified results."
    )
    cot_agents_8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    possible_answers_8 = []
    possible_thinkings_8 = []
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_sc_instruction_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking8, answer8 = await cot_agents_8[i]([taskInfo, thinking7], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_8[i].id}, calculating final sum m+n, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers_8.append(answer8)
        possible_thinkings_8.append(thinking8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + possible_thinkings_8, "Sub-task 8: Synthesize and provide the final answer m+n." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
