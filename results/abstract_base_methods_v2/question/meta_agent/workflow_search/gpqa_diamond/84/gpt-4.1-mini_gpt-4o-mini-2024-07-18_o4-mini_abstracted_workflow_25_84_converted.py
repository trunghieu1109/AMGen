async def forward_84(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Stage 1: Convert wavelength shifts and calculate velocities
    
    # Subtask 1a: Convert wavelength shifts from Angstroms to meters
    cot_instruction_1a = (
        "Sub-task 1a: Convert the observed maximum wavelength shifts of 0.03 Å for Planet1 and 0.04 Å for Planet2 "
        "to meters, ensuring unit consistency and correctness."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, converting wavelength shifts, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    # Subtask 1b: Calculate stellar reflex velocities (K_star) from wavelength shifts
    cot_instruction_1b = (
        "Sub-task 1b: Calculate the stellar reflex velocities (K_star) induced by Planet1 and Planet2 using the Doppler formula Δλ/λ = v/c, "
        "where Δλ is the wavelength shift in meters and λ = 6300 Å converted to meters. Verify that the resulting velocities are astrophysically plausible (on the order of km/s)."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, calculating stellar reflex velocities, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    # Subtask 1c: Compute orbital velocities of planets from stellar reflex velocities
    cot_instruction_1c = (
        "Sub-task 1c: Compute the orbital velocities of Planet1 and Planet2 from the stellar reflex velocities by applying the relation v_planet = (M_star / M_planet) * K_star, "
        "using the given stellar mass (1.5 M_sun) and planetary masses (7 and 5 Earth masses respectively)."
    )
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "CoT"
    }
    thinking_1c, answer_1c = await cot_agent_1c([taskInfo, thinking_1b, answer_1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, computing orbital velocities, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    # Stage 2: Extract orbital periods and calculate orbital radii
    
    # Subtask 2a: Extract or estimate orbital periods from transit data or assumptions
    cot_instruction_2a = (
        "Sub-task 2a: Extract or estimate the orbital periods of Planet1 and Planet2 from the transit method data or given context. "
        "If not explicitly provided, infer or state assumptions clearly."
    )
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_2a, answer_2a = await cot_agent_2a([taskInfo], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, extracting orbital periods, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    # Subtask 2b: Calculate orbital radii using Kepler's third law
    cot_sc_instruction_2b = (
        "Sub-task 2b: Calculate the orbital radii (semi-major axes) of Planet1 and Planet2 using Kepler's third law: a^3 = (G * M_star * P^2) / (4π^2), "
        "where P is the orbital period from Subtask 2a, M_star is the stellar mass, and G is the gravitational constant. Express results in astronomical units (AU)."
    )
    N = self.max_sc
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_2b, answer_2b = await cot_agents_2b[i]([taskInfo, thinking_2a, answer_2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, calculating orbital radii, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
        possible_answers_2b.append(answer_2b.content)
        thinkingmapping_2b[answer_2b.content] = thinking_2b
        answermapping_2b[answer_2b.content] = answer_2b
    answer_2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[answer_2b_content]
    answer_2b = answermapping_2b[answer_2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    # Subtask 2c: Consistency check of orbital radii with orbital velocities
    cot_reflect_instruction_2c = (
        "Sub-task 2c: Perform a consistency check on the orbital radii by comparing the values derived from Kepler's law with those implied by the orbital velocities from Subtask 1c (using v = 2πa / P). "
        "Confirm astrophysical plausibility and unit consistency."
    )
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2c = self.max_round
    cot_inputs_2c = [taskInfo, thinking_1c, answer_1c, thinking_2a, answer_2a, thinking_2b, answer_2b]
    subtask_desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_reflect_instruction_2c,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Reflexion"
    }
    thinking_2c, answer_2c = await cot_agent_2c(cot_inputs_2c, cot_reflect_instruction_2c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2c.id}, checking orbital radii consistency, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    for i in range(N_max_2c):
        feedback, correct = await critic_agent_2c([taskInfo, thinking_2c, answer_2c],
                                                 "please review the orbital radii consistency check and provide its limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2c.extend([thinking_2c, answer_2c, feedback])
        thinking_2c, answer_2c = await cot_agent_2c(cot_inputs_2c, cot_reflect_instruction_2c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2c.id}, refining orbital radii consistency check, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 2c: ", sub_tasks[-1])
    
    # Stage 3: Calculate equilibrium temperatures and their ratio
    
    # Subtask 3a: Calculate equilibrium temperatures of Planet1 and Planet2
    cot_reflect_instruction_3a = (
        "Sub-task 3a: Calculate the equilibrium temperatures (T_eq) of Planet1 and Planet2 using the formula T_eq = T_eff * sqrt(R_star / (2 * a)), "
        "where T_eff = 6300 K, R_star = 1.2 R_sun (converted to meters), and a is the orbital radius from Subtask 2b. Assume equal albedo and circular orbits."
    )
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3a = self.max_round
    cot_inputs_3a = [taskInfo, thinking_2b, answer_2b]
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3a, answer_3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, calculating equilibrium temperatures, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    for i in range(N_max_3a):
        feedback, correct = await critic_agent_3a([taskInfo, thinking_3a, answer_3a],
                                                 "please review the equilibrium temperature calculations and provide its limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking_3a, answer_3a, feedback])
        thinking_3a, answer_3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining equilibrium temperatures, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])
    
    # Subtask 3b: Compute the ratio of equilibrium temperatures T_eq(Planet1) / T_eq(Planet2)
    cot_instruction_3b = (
        "Sub-task 3b: Compute the ratio of the equilibrium temperatures T_eq(Planet1) / T_eq(Planet2) using the values obtained in Subtask 3a. "
        "Verify the ratio's physical plausibility."
    )
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking_3b, answer_3b = await cot_agent_3b([taskInfo, thinking_3a, answer_3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, computing temperature ratio, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])
    
    # Stage 4: Compare ratio with choices and validate output format
    
    # Subtask 4: Compare calculated temperature ratio with multiple-choice options
    cot_instruction_4 = (
        "Sub-task 4: Compare the calculated temperature ratio from Subtask 3b with the provided multiple-choice options (~1.05, ~0.98, ~0.53, ~1.30) "
        "and select the letter choice (A, B, C, or D) corresponding to the closest value."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking_4, answer_4 = await cot_agent_4([taskInfo, thinking_3b, answer_3b], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, selecting closest choice, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    # Subtask 5: Validate final output format to ensure only letter choice is returned
    debate_instruction_5 = (
        "Sub-task 5: Validate the final output format to ensure only the letter choice (A, B, C, or D) is returned, complying strictly with the query requirements."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating output format, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking5[r].append(thinking_5)
            all_answer5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the validated letter choice output.", is_sub_task=True)
    agents.append(f"Final Decision agent, validating final output format, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
