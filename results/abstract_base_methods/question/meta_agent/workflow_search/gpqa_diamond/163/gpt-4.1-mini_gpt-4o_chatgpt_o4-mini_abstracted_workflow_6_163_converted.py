async def forward_163(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Extract and organize the observational data for both binary star systems, including orbital periods and radial velocity amplitudes of each star, to prepare for subsequent calculations."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting and organizing observational data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Calculate the orbital angular velocities (omega) for system_1 and system_2 from their respective orbital periods using the formula omega = 2 * pi / P. Use the orbital periods extracted in Sub-task 1."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating orbital angular velocities, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_sc_instruction_3 = "Sub-task 3: Determine the mass ratio of the two stars in each system from their radial velocity amplitudes, using the relation M1/M2 = K2/K1, based on the radial velocity amplitudes extracted in Sub-task 1."
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, determining mass ratios, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_reflect_instruction_4a = "Sub-task 4a: Derive the total semi-major axis (a) of each binary system from the sum of the radial velocity amplitudes and the orbital period, using the relation (K1 + K2) = (2 * pi * a) / P. Show clear algebraic steps solving for a. Use data from Sub-tasks 1 and 2."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_reflect_instruction_4a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking1, answer1, thinking2, answer2], cot_reflect_instruction_4a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, deriving semi-major axis, thinking: {thinking4a.content}; answer: {answer4a.content}")
    critic_agent_4a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4a = self.max_round
    cot_inputs_4a = [taskInfo, thinking1, answer1, thinking2, answer2, thinking4a, answer4a]
    for i in range(N_max_4a):
        feedback4a, correct4a = await critic_agent_4a(cot_inputs_4a, "Critically evaluate the derivation of the semi-major axis for correctness and clarity.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4a.id}, providing feedback on semi-major axis derivation, thinking: {feedback4a.content}; answer: {correct4a.content}")
        if correct4a.content == "True":
            break
        cot_inputs_4a.extend([feedback4a])
        thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, refining semi-major axis derivation, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_reflect_instruction_4b = "Sub-task 4b: Apply Kepler's third law explicitly to calculate the total mass (M = M1 + M2) of each system using the formula M = 4 * pi^2 * a^3 / (G * P^2). Include the gravitational constant G and ensure consistent units. Use the semi-major axis derived in Sub-task 4a and orbital periods from Sub-task 1."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_reflect_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking4b, answer4b = await cot_agent_4b([taskInfo, thinking4a, answer4a, thinking1, answer1], cot_reflect_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, applying Kepler's third law for total mass, thinking: {thinking4b.content}; answer: {answer4b.content}")
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4b = self.max_round
    cot_inputs_4b = [taskInfo, thinking4a, answer4a, thinking4b, answer4b]
    for i in range(N_max_4b):
        feedback4b, correct4b = await critic_agent_4b(cot_inputs_4b, "Critically evaluate the total mass calculation using Kepler's third law for correctness and dimensional consistency.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback on total mass calculation, thinking: {feedback4b.content}; answer: {correct4b.content}")
        if correct4b.content == "True":
            break
        cot_inputs_4b.extend([feedback4b])
        thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining total mass calculation, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_reflect_instruction_4c = "Sub-task 4c: Perform a dimensional and sanity check on the derived total mass formula and intermediate results from Sub-task 4b to verify physical correctness and consistency before proceeding."
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_reflect_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking4c, answer4c = await cot_agent_4c([taskInfo, thinking4b, answer4b], cot_reflect_instruction_4c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, performing dimensional and sanity check, thinking: {thinking4c.content}; answer: {answer4c.content}")
    critic_agent_4c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4c = self.max_round
    cot_inputs_4c = [taskInfo, thinking4c, answer4c]
    for i in range(N_max_4c):
        feedback4c, correct4c = await critic_agent_4c(cot_inputs_4c, "Critically evaluate the dimensional and sanity check for correctness and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4c.id}, providing feedback on sanity check, thinking: {feedback4c.content}; answer: {correct4c.content}")
        if correct4c.content == "True":
            break
        cot_inputs_4c.extend([feedback4c])
        thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, refining sanity check, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {
        "thinking": thinking4c,
        "answer": answer4c
    }
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Compute the ratio of the total mass of system_1 to the total mass of system_2 using the masses derived in Sub-task 4b, ensuring the ratio is physically consistent and dimensionally correct."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4b, answer4b], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4b, answer4b] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing mass ratio, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the mass ratio factor.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing mass ratio factor, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_reflect_instruction_6 = "Sub-task 6: Match the computed mass ratio factor from Sub-task 5 to the closest given multiple-choice option and select the corresponding letter choice (A, B, C, or D)."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, matching mass ratio to choices, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6], "Critically evaluate the choice matching for correctness and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining choice matching, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
