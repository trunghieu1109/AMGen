async def forward_62(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    Mx = 20  # amu
    My = 2   # amu
    R_angstrom = 2  # angstroms
    omega = 4e14  # rad/s
    hbar = 1.054571817e-34  # J*s (reduced Planck constant)
    c = 3.0e8  # m/s (speed of light)
    amu_to_kg = 1.66053906660e-27  # kg
    angstrom_to_m = 1e-10  # m
    # Stage 1: Calculate reduced mass (μ)
    cot_instruction_1 = (
        "Sub-task 1: Calculate the reduced mass (μ) of the diatomic molecule using the given atomic masses Mx = 20 amu and My = 2 amu. "
        "Use μ = (Mx * My) / (Mx + My) and convert amu to kg."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculate reduced mass μ, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Extract numeric reduced mass μ from answer1 for next subtasks
    # We expect answer1.content to contain numeric value in kg
    # Stage 1: Calculate moment of inertia (I)
    cot_instruction_2 = (
        "Sub-task 2: Calculate the moment of inertia (I) of the molecule using the reduced mass (μ) from Subtask 1 and the bond length R = 2 angstroms. "
        "Use I = μ * R^2, converting R from angstroms to meters."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculate moment of inertia I, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 1: Confirm formula for rotational constant B
    cot_instruction_3a = (
        "Sub-task 3a: Confirm and explicitly state the correct formula for the rotational constant B in energy units as B = ħ² / (2I), "
        "where ħ is the reduced Planck constant (1.054571817e-34 J*s). Include units and constants explicitly."
    )
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, confirm formula for rotational constant B, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    # Stage 1: Calculate rotational constant B numerically
    cot_instruction_3b = (
        "Sub-task 3b: Calculate the rotational constant B numerically using the moment of inertia (I) from Subtask 2 and the formula B = ħ² / (2I). "
        "Express B in Joules."
    )
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking2, answer2, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, calculate rotational constant B, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    # Stage 1: Calculate vibrational energy quantum ΔEvib
    cot_instruction_4 = (
        "Sub-task 4: Calculate the vibrational energy quantum ΔEvib = ħ * ω using the given angular frequency ω = 4 × 10^14 rad/s. "
        "Express ΔEvib in Joules."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculate vibrational energy quantum ΔEvib, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Stage 2: Identify lowest energy transition
    cot_instruction_5 = (
        "Sub-task 5: Identify the lowest possible energy transition from the fundamental state to the next state with the lowest energy increase, "
        "considering vibrational and rotational energy levels. For the non-rigid rotor model, this is the first vibrational excitation plus the lowest rotational excitation (J=0 to J=1)."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3b, answer3b, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, identify lowest energy transition, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    # Stage 2: Calculate rotational energy difference ΔErot for J=0 to J=1
    cot_instruction_6a = (
        "Sub-task 6a: Calculate the rotational energy difference ΔErot for the transition J=0 to J=1 using ΔErot = 2B, "
        "where B is from Subtask 3b."
    )
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking5, answer5, thinking3b, answer3b], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, calculate rotational energy difference ΔErot, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    # Stage 2: Calculate total energy difference ΔE = ΔEvib + ΔErot
    cot_instruction_6b = (
        "Sub-task 6b: Calculate the total energy difference ΔE for the transition by summing the vibrational energy quantum ΔEvib from Subtask 4 "
        "and the rotational energy difference ΔErot from Subtask 6a."
    )
    cot_agent_6b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_instruction_6b,
        "context": ["user query", "thinking of subtask 6a", "answer of subtask 6a", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking6b, answer6b = await cot_agent_6b([taskInfo, thinking6a, answer6a, thinking4, answer4], cot_instruction_6b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6b.id}, calculate total energy difference ΔE, thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    # Stage 2: Convert ΔE to photon momentum p = ΔE / c
    cot_instruction_7a = (
        "Sub-task 7a: Convert the total energy difference ΔE from Subtask 6b into the photon momentum p using the relation p = ΔE / c, "
        "where c = 3.0 × 10^8 m/s. Ensure unit consistency."
    )
    cot_agent_7a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7a = {
        "subtask_id": "subtask_7a",
        "instruction": cot_instruction_7a,
        "context": ["user query", "thinking of subtask 6b", "answer of subtask 6b"],
        "agent_collaboration": "CoT"
    }
    thinking7a, answer7a = await cot_agent_7a([taskInfo, thinking6b, answer6b], cot_instruction_7a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7a.id}, convert ΔE to photon momentum p, thinking: {thinking7a.content}; answer: {answer7a.content}")
    sub_tasks.append(f"Sub-task 7a output: thinking - {thinking7a.content}; answer - {answer7a.content}")
    subtask_desc7a['response'] = {"thinking": thinking7a, "answer": answer7a}
    logs.append(subtask_desc7a)
    print("Step 7a: ", sub_tasks[-1])
    # Stage 2: Reflexion step to verify photon momentum p
    cot_reflect_instruction_7b = (
        "Sub-task 7b: Perform a reflexion step to verify the physical plausibility, unit consistency, and correctness of the calculated photon momentum p from Subtask 7a. "
        "Cross-check intermediate values and formulas to detect errors early."
    )
    cot_agent_7b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7b = self.max_round
    cot_inputs_7b = [taskInfo, thinking7a, answer7a]
    thinking7b, answer7b = await cot_agent_7b(cot_inputs_7b, cot_reflect_instruction_7b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7b.id}, reflexion on photon momentum p, thinking: {thinking7b.content}; answer: {answer7b.content}")
    for i in range(N_max_7b):
        feedback, correct = await critic_agent_7b([taskInfo, thinking7b, answer7b],
                                                 "Review the photon momentum calculation for physical plausibility, unit consistency, and formula correctness.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7b.extend([thinking7b, answer7b, feedback])
        thinking7b, answer7b = await cot_agent_7b(cot_inputs_7b, cot_reflect_instruction_7b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7b.id}, refining photon momentum p, thinking: {thinking7b.content}; answer: {answer7b.content}")
    sub_tasks.append(f"Sub-task 7b output: thinking - {thinking7b.content}; answer - {answer7b.content}")
    subtask_desc7b = {
        "subtask_id": "subtask_7b",
        "instruction": cot_reflect_instruction_7b,
        "context": ["user query", "thinking of subtask 7a", "answer of subtask 7a"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc7b['response'] = {"thinking": thinking7b, "answer": answer7b}
    logs.append(subtask_desc7b)
    print("Step 7b: ", sub_tasks[-1])
    # Stage 3: Compare calculated photon momentum p with multiple-choice options
    debate_instruction_8 = (
        "Sub-task 8: Compare the calculated photon momentum p from Subtask 7b with the given multiple-choice options. "
        "Evaluate absolute differences and orders of magnitude, then select the option closest in value and physically consistent with the calculation."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": debate_instruction_8,
        "context": ["user query", "thinking of subtask 7b", "answer of subtask 7b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking7b, answer7b], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking7b, answer7b] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct photon momentum, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make final decision on the correct photon momentum choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final photon momentum, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
