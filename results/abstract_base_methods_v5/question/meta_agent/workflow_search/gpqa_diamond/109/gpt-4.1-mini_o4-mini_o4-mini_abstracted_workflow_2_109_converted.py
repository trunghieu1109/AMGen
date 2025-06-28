async def forward_109(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Stage 1: Calculate and verify apparent magnitudes for all stars
    
    # Sub-task 1a: Calculate apparent V magnitudes for Star1, Star4, Star5 (absolute magnitudes)
    cot_instruction_1a = (
        "Sub-task 1a: For Star1, Star4, and Star5, calculate their apparent V magnitudes using the distance modulus formula "
        "and add extinction corrections where E(B-V) is provided. Mark these magnitudes as computed_apparent. "
        "Use the relation: apparent_mag = absolute_mag + 5*log10(distance_pc) - 5 + extinction (if any). "
        "Clearly indicate which stars have extinction applied."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, calculating apparent magnitudes for Star1, Star4, Star5, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    # Sub-task 1b: Accept given apparent magnitudes for Star2 and Star3 as including extinction, mark as given_apparent
    cot_instruction_1b = (
        "Sub-task 1b: For Star2 and Star3, accept the provided apparent V magnitudes as already including extinction. "
        "Do not apply any further extinction correction. Mark these magnitudes as given_apparent. "
        "Explain the reasoning to avoid double extinction correction, especially for Star3."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, accepting given apparent magnitudes for Star2 and Star3, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    # Sub-task 1c: Verify and cross-check apparent magnitudes from 1a and 1b to ensure no double extinction correction
    cot_sc_instruction_1c = (
        "Sub-task 1c: Verify and cross-check the apparent magnitudes obtained in subtasks 1a and 1b to ensure no double extinction correction is applied, "
        "especially for Star3. Use self-consistency by generating multiple reasoning paths: one assuming given apparent magnitudes include extinction, "
        "another assuming they do not, and compare results to confirm the correct interpretation."
    )
    N = self.max_sc
    cot_agents_1c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_sc_instruction_1c,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1c, answer1c = await cot_agents_1c[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_sc_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, verifying apparent magnitudes consistency, thinking: {thinking1c.content}; answer: {answer1c.content}")
        possible_answers_1c.append(answer1c.content)
        thinkingmapping_1c[answer1c.content] = thinking1c
        answermapping_1c[answer1c.content] = answer1c
    answer1c_content = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking1c = thinkingmapping_1c[answer1c_content]
    answer1c = answermapping_1c[answer1c_content]
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking1c, "answer": answer1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    # Stage 2: Determine stars meeting detection limits and identify observatory declination coverage
    
    # Sub-task 2: Determine which stars meet detection limits of ESPRESSO (V<17) and HIRES (V<16) using verified magnitudes
    cot_reflect_instruction_2 = (
        "Sub-task 2: Determine which stars meet the apparent V magnitude detection limits for ESPRESSO (V < 17 mag) "
        "and HIRES (V < 16 mag) based on the verified apparent magnitudes from Sub-task 1c."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking1c, answer1c]
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, filtering stars by detection limits, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2],
                                                "Please review the filtering of stars by detection limits and provide limitations.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining filtering of stars, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Identify declination coverage ranges of Paranal and Keck observatories
    cot_instruction_3 = (
        "Sub-task 3: Identify the declination coverage ranges of the Paranal Observatory (ESPRESSO) and the Keck Observatory (HIRES) "
        "from the provided links, ignoring pointing and altitude limits, to establish which stars are observable from both sites."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, identifying observatory declination coverage, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 3: Cross-reference stars by magnitude and declination, debate borderline cases, and finalize selection
    
    # Sub-task 4: Cross-reference stars meeting magnitude limits with declination coverage to find stars observable by both
    cot_instruction_4 = (
        "Sub-task 4: Cross-reference stars that satisfy both spectrographs' magnitude limits (from Sub-task 2) "
        "with the declination coverage of both observatories (from Sub-task 3) to identify stars observable by both ESPRESSO and HIRES."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, cross-referencing stars by magnitude and declination, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Debate borderline cases to confirm inclusion/exclusion and ensure robust final selection
    debate_instruction_5 = (
        "Sub-task 5: Conduct a debate on borderline cases (e.g., stars near magnitude or declination limits) to confirm their inclusion or exclusion, "
        "ensuring a robust final selection of stars observable by both ESPRESSO and HIRES."
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
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating borderline stars, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on stars observable by both ESPRESSO and HIRES.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on observable stars, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    
    # Sub-task 6: Compare final star list with multiple-choice options and select correct choice or state none match
    cot_instruction_6 = (
        "Sub-task 6: Compare the final list of stars observable by both spectrographs against the provided multiple-choice options (A-D). "
        "If no option matches exactly, explicitly state that none of the provided options are correct and justify this conclusion."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting correct multiple-choice option or none, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc_6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
