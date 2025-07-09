async def forward_186(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Gather all ESPRESSO+VLT instrument parameters needed for S/N calculations: telescope collecting area (m²), spectral resolution (R), total system throughput, V-band pixel/bin width (Å), detector quantum efficiency, and detector noise characteristics."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, gathering instrument parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction2 = "Sub-task 2: Retrieve Paranal Observatory site parameters relevant for observability: latitude, elevation limits, maximum usable airmass, and declination range for routine observations."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, retrieving site parameters, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction3 = "Sub-task 3: Obtain coordinates (RA, DEC), V-band apparent magnitudes, and distances for Canopus and Polaris from astronomical catalogs or literature."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers3 = []
    thinking_map3 = {}
    answer_map3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction3,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents3:
        t3, a3 = await agent([taskInfo], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, retrieving star data, thinking: {t3.content}; answer: {a3.content}")
        possible_answers3.append(a3.content)
        thinking_map3[a3.content] = t3
        answer_map3[a3.content] = a3
    answer3_content = Counter(possible_answers3).most_common(1)[0][0]
    thinking3 = thinking_map3[answer3_content]
    answer3 = answer_map3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction4 = "Sub-task 4: Compute the apparent V magnitude of each artificial star with Absolute V = 15 mag at distances of 5, 10, 50, and 200 pc using the distance modulus formula."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, computing apparent magnitudes, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction5 = "Sub-task 5: Compute the incident photon flux at the telescope aperture for each star: define the V-band zero-point flux in photons/s/m²/Å, and use each star’s apparent V magnitude, telescope area, and throughput to calculate photons/s/Å."
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers5 = []
    thinking_map5 = {}
    answer_map5 = {}
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction5,
        "context": ["user query", "thinking1", "answer1", "thinking3", "answer3", "thinking4", "answer4"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents5:
        t5, a5 = await agent([taskInfo, thinking1, answer1, thinking3, answer3, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, converting magnitudes to flux, thinking: {t5.content}; answer: {a5.content}")
        possible_answers5.append(a5.content)
        thinking_map5[a5.content] = t5
        answer_map5[a5.content] = a5
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinking_map5[answer5_content]
    answer5 = answer_map5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction6 = "Sub-task 6: Calculate the total number of photons per spectral bin in a 1-hour exposure and derive the S/N for each star assuming photon-noise dominance."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction6,
        "context": ["user query", "thinking5", "answer5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, calculating photons per bin and S/N, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_sc_instruction7 = "Sub-task 7: Perform a self-consistency check by independently repeating the S/N calculation for a subset of stars and ensure numerical agreement within tolerance."
    N7 = self.max_sc
    cot_agents7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N7)]
    possible_answers7 = []
    thinking_map7 = {}
    answer_map7 = {}
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction7,
        "context": ["user query", "thinking6", "answer6"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents7:
        t7, a7 = await agent([taskInfo, thinking6, answer6], cot_sc_instruction7, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, repeating S/N calculation, thinking: {t7.content}; answer: {a7.content}")
        possible_answers7.append(a7.content)
        thinking_map7[a7.content] = t7
        answer_map7[a7.content] = a7
    answer7_content = Counter(possible_answers7).most_common(1)[0][0]
    thinking7 = thinking_map7[answer7_content]
    answer7 = answer_map7[answer7_content]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction8 = "Sub-task 8: Evaluate each star’s observability from Paranal: compute its maximum elevation and corresponding airmass at meridian transit, then flag stars with airmass ≤ maximum usable airmass and DEC within allowable range."
    cot_agent8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction8,
        "context": ["user query", "thinking2", "answer2", "thinking3", "answer3"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent8([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, checking observability, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_reflect_instruction9 = "Sub-task 9: Combine the S/N flags (S/N ≥ 10) from Subtasks 6 & 7 with the observability flags from Subtask 8 to classify each star as 'detectable' or 'not detectable'."
    cot_agent9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs9 = [taskInfo, thinking6, answer6, thinking7, answer7, thinking8, answer8]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction9,
        "context": ["user query", "thinking6", "answer6", "thinking7", "answer7", "thinking8", "answer8"],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent9(cot_inputs9, cot_reflect_instruction9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent9.id}, classifying detectability, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(self.max_round):
        feedback9, correct9 = await critic_agent9([taskInfo, thinking9, answer9], "Please review the classification and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent9.id}, providing feedback, feedback: {feedback9.content}; correct: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs9.extend([thinking9, answer9, feedback9])
        thinking9, answer9 = await cot_agent9(cot_inputs9, cot_reflect_instruction9, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent9.id}, refining classification, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    cot_instruction10 = "Sub-task 10: Count the number of stars classified as 'detectable' in Subtask 9 and map that count to the provided multiple-choice options: 2 (A), 3 (B), 4 (C), or 5 (D)."
    cot_agent10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction10,
        "context": ["user query", "thinking9", "answer9"],
        "agent_collaboration": "CoT"
    }
    thinking10, answer10 = await cot_agent10([taskInfo, thinking9, answer9], cot_instruction10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent10.id}, counting detectable stars and mapping to choice, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs