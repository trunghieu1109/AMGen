async def forward_51(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify and extract all given physical parameters of the star relevant to the problem, including radius, mass, effective temperatures (with and without spots), spot coverage fraction, and the wavelength corresponding to the energy level transition."
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
    
    cot_instruction_2 = "Sub-task 2: Understand and restate the physical context and assumptions: the star is in Local Thermodynamic Equilibrium (LTE), the photospheric effective temperature changes due to spot coverage, and the observed quantity is the ratio of neutral Ti atoms in two energy levels."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, restating physical context, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Calculate the energy difference (ΔE) between the two Ti atomic levels using the given wavelength (1448 Å) of the transition, applying the relation ΔE = hc/λ, and explicitly compute ΔE/k (energy difference divided by Boltzmann constant) for later use."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating energy difference ΔE and ΔE/k, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Formulate the expression for the ratio of the number of neutral Ti atoms in the two energy levels under LTE using the Boltzmann distribution, explicitly showing dependence on temperature and energy difference ΔE/k."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, formulating Boltzmann ratio expression, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5 = "Sub-task 5: Model the star's photosphere as a composite of spotted and unspotted regions: calculate the spot temperature assuming 40% coverage and given overall effective temperature (5500 K), using the relation Teff^4 = f * Tspot^4 + (1-f) * Tnonspot^4, where f is spot coverage fraction."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating spot temperature from composite photosphere model, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6 = "Sub-task 6: Calculate the Boltzmann population ratios (level 1 to level 2) separately for the unspotted region at 6000 K and the spotted region at the spot temperature from subtask_5, using the expression from subtask_4 and the numeric ΔE/k from subtask_3. Decompose this into explicit numeric steps: compute ΔE/k, compute Δ(1/T), calculate exponent product, and evaluate exponential factor for each temperature."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3", "thinking of subtask_4", "answer of subtask_4", "thinking of subtask_5", "answer of subtask_5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating Boltzmann population ratios for spotted and unspotted regions, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Compute the weighted average population ratio for the composite photosphere by combining the ratios from subtask_6 weighted by their respective surface coverage fractions (60% unspotted, 40% spotted)."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask_6", "answer of subtask_6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, computing weighted average population ratio for composite photosphere, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Sub-task 8: Calculate the population ratio for the star without spots at 6000 K using the Boltzmann expression from subtask_4 and numeric ΔE/k from subtask_3."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, calculating population ratio for star without spots, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {
        "thinking": thinking8,
        "answer": answer8
    }
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    cot_instruction_9 = "Sub-task 9: Determine the factor by which the ratio of neutral Ti atoms in the two energy levels changes when the star does not have spots compared to when it has spots by dividing the ratio from subtask_8 by the weighted average ratio from subtask_7. Decompose the calculation into explicit numeric steps to ensure accuracy."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", "thinking of subtask_7", "answer of subtask_7", "thinking of subtask_8", "answer of subtask_8"],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking7, answer7, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, calculating factor change of population ratio, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {
        "thinking": thinking9,
        "answer": answer9
    }
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    cot_instruction_10 = "Sub-task 10: Compare the calculated factor from subtask_9 with the provided multiple-choice options (~2.9, ~4.5, ~7.8, ~1.1) and select the closest matching choice."
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction_10,
        "context": ["user query", "thinking of subtask_9", "answer of subtask_9"],
        "agent_collaboration": "CoT"
    }
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking9, answer9], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, selecting closest multiple-choice answer, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {
        "thinking": thinking10,
        "answer": answer10
    }
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    
    cot_reflect_instruction_11 = "Sub-task 11: Perform a reflexion and verification step: cross-check numeric calculations and physical assumptions, confirm that the composite photosphere model and numeric propagation are consistent, and validate that the selected choice is physically reasonable and matches the computed factor."
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_11 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_11 = self.max_round
    cot_inputs_11 = [taskInfo, thinking9, answer9, thinking10, answer10]
    thinking11, answer11 = await cot_agent_11(cot_inputs_11, cot_reflect_instruction_11, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_11.id}, performing reflexion and verification, thinking: {thinking11.content}; answer: {answer11.content}")
    for i in range(N_max_11):
        feedback, correct = await critic_agent_11([taskInfo, thinking11, answer11], "Please review the numeric calculations, physical assumptions, and final choice for consistency and correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_11.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_11.extend([thinking11, answer11, feedback])
        thinking11, answer11 = await cot_agent_11(cot_inputs_11, cot_reflect_instruction_11, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_11.id}, refining reflexion and verification, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_reflect_instruction_11,
        "context": ["user query", "thinking of subtask_9", "answer of subtask_9", "thinking of subtask_10", "answer of subtask_10"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc11['response'] = {
        "thinking": thinking11,
        "answer": answer11
    }
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer, logs
