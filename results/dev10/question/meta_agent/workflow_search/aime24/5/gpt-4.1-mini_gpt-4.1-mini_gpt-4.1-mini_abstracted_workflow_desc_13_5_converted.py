async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # --------------------------------------------------------------------------------------------------------------
    # Stage 1: Geometry Analysis and Basic Computations
    # Objective: Analyze tetrahedron, compute face area once with symmetry, compute volume with Cayley–Menger determinant
    # Agent Collaboration: CoT and SC_CoT

    # Sub-task 1: Analyze and represent the tetrahedron with given edge lengths and symmetry
    cot_instruction_1 = (
        "Sub-task 1: Analyze the tetrahedron ABCD with edges AB=CD=\u221A41, AC=BD=\u221A80, BC=AD=\u221A89. "
        "Verify triangle inequalities for all faces to ensure non-degeneracy. Identify symmetry from equal opposite edges. "
        "Construct a coordinate or vector representation respecting these symmetries without assuming angles. "
        "Document all assumptions clearly."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1, answer_1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing tetrahedron geometry, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)

    # Sub-task 2: Compute the area of one triangular face using Heron's formula once, enforce symmetry
    cot_sc_instruction_2 = (
        "Sub-task 2: Compute the area of one triangular face with sides \u221A41, \u221A80, \u221A89 using Heron's formula symbolically. "
        "Since opposite edges are equal, all four faces are congruent and have the same area. "
        "Perform numeric approximation to verify consistency and rule out errors. "
        "Explicitly assert symmetry and congruency of all faces."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    possible_thinkings_2 = []
    for i in range(N_sc):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1.content, answer_1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, computing face area, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent and correct face area calculation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)

    # Sub-task 3: Calculate volume using Cayley–Menger determinant, symbolic and numeric cross-validation
    cot_sc_instruction_3 = (
        "Sub-task 3: Calculate the volume of tetrahedron ABCD using the Cayley–Menger determinant with given edges. "
        "Maintain exact radical expressions symbolically and perform numeric approximation for cross-validation. "
        "Compare with volume from triple scalar product using coordinates from Sub-task 1 to ensure consistency. "
        "Confirm volume is positive and consistent with geometry."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3 = []
    possible_thinkings_3 = []
    for i in range(N_sc):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_1.content, answer_1.content], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, computing volume, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3)
        possible_thinkings_3.append(thinking_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent and correct volume calculation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)

    print("Step 1: ", sub_tasks[-3])
    print("Step 2: ", sub_tasks[-2])
    print("Step 3: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 2: Tangentiality Verification and Inradius Calculation
    # Objective: Verify existence of inscribed sphere (tangentiality), then compute inradius
    # Agent Collaboration: CoT, Debate, Reflexion, SC_CoT

    # Sub-task 1: Verify tangentiality and existence of incenter using face areas and volume
    cot_debate_instruction_4 = (
        "Sub-task 4: Verify if tetrahedron ABCD is tangential, i.e., admits an inscribed sphere with a unique incenter. "
        "Use computed face areas and volume to check necessary and sufficient conditions for tangentiality. "
        "Confirm that a point equidistant from all faces exists. "
        "If tangentiality fails, flag error and halt further inradius calculation. "
        "Document all verification steps explicitly."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_debate_instruction_4,
        "context": ["user query", thinking_2.content, answer_2.content, thinking_3.content, answer_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_2.content, answer_2.content, thinking_3.content, answer_3.content], cot_debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_2.content, answer_2.content, thinking_3.content, answer_3.content] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, cot_debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying tangentiality, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Final decision on tangentiality verification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)

    print("Step 4: ", sub_tasks[-1])

    # Sub-task 2: Calculate inradius r = 3 * volume / surface area if tangentiality confirmed
    cot_sc_instruction_5 = (
        "Sub-task 5: Given tangentiality confirmation, calculate the inradius r = 3 * volume / surface area. "
        "Use exact symbolic expressions and simplify to form (m * sqrt(n)) / p, with m, p positive coprime integers and n square-free. "
        "Perform numeric approximations alongside symbolic manipulations to verify correctness. "
        "Avoid premature rounding or simplification that obscures final form."
    )
    if "not" in answer_4.content.lower() or "no" in answer_4.content.lower():
        raise ValueError("Tangentiality verification failed. Cannot compute inradius.")
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_5 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking_2.content, answer_2.content, thinking_3.content, answer_3.content, thinking_4.content, answer_4.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_5 = []
    possible_thinkings_5 = []
    for i in range(N_sc):
        thinking_5, answer_5 = await cot_agents_5[i]([taskInfo, thinking_2.content, answer_2.content, thinking_3.content, answer_3.content, thinking_4.content, answer_4.content], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, calculating inradius, thinking: {thinking_5.content}; answer: {answer_5.content}")
        possible_answers_5.append(answer_5)
        possible_thinkings_5.append(thinking_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + possible_answers_5 + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent and correct inradius calculation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)

    print("Step 5: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------
    # Stage 3: Final Verification and Result Computation
    # Objective: Verify inradius expression, coprimality, square-freeness, and compute m+n+p
    # Agent Collaboration: SC_CoT, Reflexion, Debate

    reflect_inst_6 = (
        "Sub-task 6: Perform comprehensive verification of the inradius calculation and existence of point I inside tetrahedron. "
        "Confirm final expression matches symbolic and numeric results. Verify m and p are coprime, n is square-free. "
        "Check consistency with tetrahedron geometry and no contradictions from earlier assumptions. "
        "Compute and return the sum m + n + p as the final answer. "
        "Provide detailed explanation of verification steps and conclusion. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking_5.content, answer_5.content, thinking_4.content, answer_4.content]
    subtask_desc_6 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": reflect_inst_6,
        "context": ["user query", thinking_5.content, answer_5.content, thinking_4.content, answer_4.content],
        "agent_collaboration": "SC_CoT | Reflexion | Debate"
    }
    thinking_6, answer_6 = await cot_agent_6(cot_inputs_6, reflect_inst_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, verifying inradius and final result, thinking: {thinking_6.content}; answer: {answer_6.content}")
    for i in range(N_max_6):
        feedback_6, correct_6 = await critic_agent_6([taskInfo, thinking_6.content, answer_6.content], "Please review and provide limitations of provided solutions. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback_6.content}; answer: {correct_6.content}")
        if correct_6.content.strip() == "True":
            break
        cot_inputs_6.extend([thinking_6.content, answer_6.content, feedback_6.content])
        thinking_6, answer_6 = await cot_agent_6(cot_inputs_6, reflect_inst_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining verification, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)

    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_6, answer_6, sub_tasks, agents)
    return final_answer, logs
