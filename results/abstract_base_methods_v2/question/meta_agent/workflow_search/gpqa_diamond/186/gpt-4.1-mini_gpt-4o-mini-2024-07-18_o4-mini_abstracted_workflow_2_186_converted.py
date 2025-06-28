async def forward_186(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Extract and list all stars from the query, including their names, coordinates (RA, DEC), absolute V magnitudes, and distances, ensuring all given parameters are accurately captured."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracted stars, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2 = "Sub-task 2: Retrieve and summarize quantitative technical specifications and performance parameters of the ESPRESSO spectrograph coupled with an 8m VLT telescope relevant to S/N estimation, including limiting magnitudes for S/N=10 in 1 hour, instrument throughput, pixel sampling, read noise, sky background, and any sensitivity curves available from the official ESPRESSO documentation."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, gathered ESPRESSO specs, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Explicitly define the detectability criterion as achieving a signal-to-noise ratio (S/N) of at least 10 per binned pixel during a 1-hour exposure with ESPRESSO on an 8m VLT telescope, and clarify how this criterion will be applied in subsequent subtasks."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, defined detectability criterion, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Obtain or calculate the apparent V magnitude for each star: for known stars (Canopus and Polaris), retrieve accurate apparent magnitudes and distances from reliable astronomical catalogs (e.g., SIMBAD); for hypothetical stars, calculate apparent magnitudes using the distance modulus formula based on their absolute magnitudes and distances."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, obtained/calculated apparent magnitudes, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5a = "Sub-task 5a: Convert the apparent V magnitudes of each star into corresponding flux values using standard photometric relations, preparing inputs for S/N calculation."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4, answer4], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, converted apparent magnitudes to flux, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_instruction_5b = "Sub-task 5b: Using the ESPRESSO instrument parameters from subtask_2 and the flux values from subtask_5a, calculate the expected signal-to-noise ratio (S/N) per binned pixel for each star during a 1-hour exposure, applying quantitative models that incorporate throughput, noise sources, and pixel sampling."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_5a", "answer of subtask_5a"],
        "agent_collaboration": "CoT"
    }
    thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking2, answer2, thinking5a, answer5a], cot_instruction_5b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5b.id}, calculated expected S/N, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_reflect_instruction_5c = "Sub-task 5c: Perform a Reflexion step by generating multiple S/N estimates using different reasonable assumptions or models (e.g., varying throughput or noise parameters) to identify and reconcile any discrepancies or borderline cases in S/N calculations."
    cot_agent_5c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5c = self.max_round
    cot_inputs_5c = [taskInfo, thinking5b, answer5b]
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_reflect_instruction_5c,
        "context": ["user query", "thinking of subtask_5b", "answer of subtask_5b"],
        "agent_collaboration": "Reflexion"
    }
    thinking5c, answer5c = await cot_agent_5c(cot_inputs_5c, cot_reflect_instruction_5c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5c.id}, initial multiple S/N estimates, thinking: {thinking5c.content}; answer: {answer5c.content}")
    for i in range(N_max_5c):
        feedback, correct = await critic_agent_5c([taskInfo, thinking5c, answer5c],
                                                 "Please review the multiple S/N estimates, identify discrepancies or borderline cases, and provide limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5c.extend([thinking5c, answer5c, feedback])
        thinking5c, answer5c = await cot_agent_5c(cot_inputs_5c, cot_reflect_instruction_5c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5c.id}, refining S/N estimates, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {"thinking": thinking5c, "answer": answer5c}
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    cot_instruction_6 = "Sub-task 6: Cross-validate the calculated S/N values against published ESPRESSO sensitivity data and limiting magnitudes to detect inconsistencies or outliers, and flag any stars with uncertain detectability status for further review."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask_5c", "answer of subtask_5c"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5c, answer5c], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, cross-validated S/N values, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Determine for each star whether it meets the detectability criterion (S/N ≥ 10 per binned pixel in 1 hour) based on the validated S/N values, explicitly justifying borderline or uncertain cases with reference to instrument limits and calculations."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3", "thinking of subtask_6", "answer of subtask_6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking3, answer3, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, determined detectability per star, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Sub-task 8: Count the total number of stars that satisfy the detectability criterion based on subtask_7 results."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask_7", "answer of subtask_7"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, counted detectable stars, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    debate_instruction_9 = "Sub-task 9: Match the count of detectable stars to the provided multiple-choice options and select the correct alphabet choice (A, B, C, or D), providing a rationale that references the detectability threshold and instrument capabilities."
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": debate_instruction_9,
        "context": ["user query", "thinking of subtask_8", "answer of subtask_8"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking8, answer8], debate_instruction_9, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct choice, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    cot_reflect_instruction_10 = "Sub-task 10: Perform a final Reflexion step to rationale-check the selected answer against known ESPRESSO magnitude limits (approximately V=12–13 for S/N=10 in 1 hour) and the physical plausibility of the results, ensuring no contradictions or errors remain before finalizing the output."
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_10 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_10 = self.max_round
    cot_inputs_10 = [taskInfo, thinking9, answer9]
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_reflect_instruction_10,
        "context": ["user query", "thinking of subtask_9", "answer of subtask_9"],
        "agent_collaboration": "Reflexion"
    }
    thinking10, answer10 = await cot_agent_10(cot_inputs_10, cot_reflect_instruction_10, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_10.id}, initial rationale check, thinking: {thinking10.content}; answer: {answer10.content}")
    for i in range(N_max_10):
        feedback, correct = await critic_agent_10([taskInfo, thinking10, answer10],
                                                 "Please review the final selected answer for consistency with ESPRESSO magnitude limits and physical plausibility.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_10.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_10.extend([thinking10, answer10, feedback])
        thinking10, answer10 = await cot_agent_10(cot_inputs_10, cot_reflect_instruction_10, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_10.id}, refining rationale check, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
