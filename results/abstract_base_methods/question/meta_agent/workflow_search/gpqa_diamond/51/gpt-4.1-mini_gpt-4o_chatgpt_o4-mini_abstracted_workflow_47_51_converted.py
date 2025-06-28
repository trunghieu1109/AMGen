async def forward_51(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify and extract all given physical parameters of the star relevant to the problem, including radius, mass, effective temperatures with and without spots, spot coverage fraction, and the transition wavelength between the two Ti energy levels."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting physical parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Understand and restate the physical context and assumptions: confirm the LTE condition, interpret the meaning of effective temperature changes due to spot coverage, and clarify the significance of the Ti neutral atom level population ratio in this context."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, restating physical context, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_instruction_3 = "Sub-task 3: Formulate the expression for the ratio of the number of neutral Ti atoms in the two energy levels under LTE using the Boltzmann distribution, explicitly incorporating the energy difference corresponding to the given wavelength."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, formulating Boltzmann ratio expression, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Calculate the energy difference (ΔE) between the two Ti energy levels from the given wavelength (1448 Å) using the relation ΔE = hc/λ, and express ΔE in electronvolts (eV) and in Kelvin units (ΔE/k)."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating energy difference ΔE, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5a = "Sub-task 5a: Calculate ΔE/k in Kelvin units from the energy difference ΔE obtained in subtask 4, showing all intermediate numerical values and units clearly."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4, answer4], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, calculating ΔE/k in Kelvin units, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_instruction_5b = "Sub-task 5b: Compute the reciprocal temperatures (1/T) for both effective temperatures (6000 K without spots and 5500 K with spots), showing all intermediate values with units."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "CoT"
    }
    thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking5a, answer5a], cot_instruction_5b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5b.id}, computing reciprocal temperatures, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_instruction_5c = "Sub-task 5c: Calculate the difference between the reciprocal temperatures (1/5500 K - 1/6000 K), explicitly showing the sign and numerical values."
    cot_agent_5c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_instruction_5c,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "CoT"
    }
    thinking5c, answer5c = await cot_agent_5c([taskInfo, thinking5b, answer5b], cot_instruction_5c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5c.id}, calculating reciprocal temperature difference, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {
        "thinking": thinking5c,
        "answer": answer5c
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    cot_instruction_5d = "Sub-task 5d: Multiply ΔE/k by the temperature difference calculated in subtask 5c to obtain the exponent for the Boltzmann factor, showing all intermediate steps and units."
    cot_agent_5d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_instruction_5d,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "CoT"
    }
    thinking5d, answer5d = await cot_agent_5d([taskInfo, thinking5a, answer5a, thinking5c, answer5c], cot_instruction_5d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5d.id}, calculating exponent for Boltzmann factor, thinking: {thinking5d.content}; answer: {answer5d.content}")
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {
        "thinking": thinking5d,
        "answer": answer5d
    }
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])
    
    cot_instruction_5e = "Sub-task 5e: Calculate the exponential of the exponent obtained in subtask 5d to find the factor by which the population ratio changes between the two temperatures, showing the calculation and verifying the physical plausibility of the result."
    cot_agent_5e = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5e = {
        "subtask_id": "subtask_5e",
        "instruction": cot_instruction_5e,
        "context": ["user query", "thinking of subtask 5d", "answer of subtask 5d"],
        "agent_collaboration": "CoT"
    }
    thinking5e, answer5e = await cot_agent_5e([taskInfo, thinking5d, answer5d], cot_instruction_5e, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5e.id}, calculating exponential factor and verifying plausibility, thinking: {thinking5e.content}; answer: {answer5e.content}")
    sub_tasks.append(f"Sub-task 5e output: thinking - {thinking5e.content}; answer - {answer5e.content}")
    subtask_desc5e['response'] = {
        "thinking": thinking5e,
        "answer": answer5e
    }
    logs.append(subtask_desc5e)
    print("Step 5e: ", sub_tasks[-1])
    
    debate_instruction_5f = "Sub-task 5f: Perform a self-consistency check by independently recalculating the factor from subtasks 5a-5e and cross-verify results to ensure numerical accuracy and physical plausibility, correcting any discrepancies found."
    debate_agents_5f = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5f = self.max_round
    all_thinking5f = [[] for _ in range(N_max_5f)]
    all_answer5f = [[] for _ in range(N_max_5f)]
    subtask_desc5f = {
        "subtask_id": "subtask_5f",
        "instruction": debate_instruction_5f,
        "context": ["user query", "thinking of subtask 5e", "answer of subtask 5e"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5f):
        for i, agent in enumerate(debate_agents_5f):
            if r == 0:
                thinking5f, answer5f = await agent([taskInfo, thinking5e, answer5e], debate_instruction_5f, r, is_sub_task=True)
            else:
                input_infos_5f = [taskInfo, thinking5e, answer5e] + all_thinking5f[r-1] + all_answer5f[r-1]
                thinking5f, answer5f = await agent(input_infos_5f, debate_instruction_5f, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, performing self-consistency check, thinking: {thinking5f.content}; answer: {answer5f.content}")
            all_thinking5f[r].append(thinking5f)
            all_answer5f[r].append(answer5f)
    final_decision_agent_5f = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5f, answer5f = await final_decision_agent_5f([taskInfo] + all_thinking5f[-1] + all_answer5f[-1], "Sub-task 5f: Make final decision on the verified factor by which the population ratio changes.", is_sub_task=True)
    agents.append(f"Final Decision agent, verifying factor, thinking: {thinking5f.content}; answer: {answer5f.content}")
    sub_tasks.append(f"Sub-task 5f output: thinking - {thinking5f.content}; answer - {answer5f.content}")
    subtask_desc5f['response'] = {
        "thinking": thinking5f,
        "answer": answer5f
    }
    logs.append(subtask_desc5f)
    print("Step 5f: ", sub_tasks[-1])
    
    cot_instruction_6 = "Sub-task 6: Compare the verified calculated factor from subtask 5f with the provided multiple-choice options and select the closest matching choice."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5f", "answer of subtask 5f"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5f, answer5f], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting closest matching choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs