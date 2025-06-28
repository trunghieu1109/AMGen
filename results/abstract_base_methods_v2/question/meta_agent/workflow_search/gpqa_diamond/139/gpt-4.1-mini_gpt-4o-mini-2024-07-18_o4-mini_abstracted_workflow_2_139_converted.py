async def forward_139(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_1a = (
        "Sub-task 1a: Analyze isotopic clues in the query to generate multiple candidate substances for Substance X, "
        "focusing on the presence of a heavier isotope of one constituent element and the implication of isotopic composition."
    )
    N = self.max_sc
    cot_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_sc_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1a, answer1a = await cot_agents_1a[i]([taskInfo], cot_sc_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1a[i].id}, generating candidate substances for Substance X, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a
    answer1a_content = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[answer1a_content]
    answer1a = answermapping_1a[answer1a_content]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_sc_instruction_1b = (
        "Sub-task 1b: Evaluate reaction clues — violent reaction with liquid Y, gas W properties (molecule with equal protons and neutrons), "
        "precipitate G formation, and melting point of B near 277 K — to narrow down candidate substances for Substance X, using output from Sub-task 1a."
    )
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1b, answer1b = await cot_agents_1b[i]([taskInfo, thinking1a, answer1a], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, narrowing candidates for Substance X, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b
    answer1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[answer1b_content]
    answer1b = answermapping_1b[answer1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = (
        "Sub-task 1c: Cross-validate candidate substances for Substance X with known organic chemistry reagents and their isotopic and reaction properties, "
        "including analog reagents, to finalize the identification of Substance X, using output from Sub-task 1b."
    )
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "CoT"
    }
    thinking1c, answer1c = await cot_agent_1c([taskInfo, thinking1b, answer1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, finalizing identification of Substance X, thinking: {thinking1c.content}; answer: {answer1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc1c['response'] = {"thinking": thinking1c, "answer": answer1c}
    logs.append(subtask_desc1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Independently identify gas W using the clue that its molecule contains equal numbers of protons and neutrons, "
        "ensuring this identification is consistent with the reaction context and informs subsequent steps, using output from Sub-task 1a."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1a, answer1a], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying gas W, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = (
        "Sub-task 3: Identify liquid Y and precipitate G by interpreting the reaction context, the nature of gas W, "
        "and the formation of precipitate G that releases B upon heating, validating consistency with known chemical behavior, "
        "using outputs from Sub-task 1c and Sub-task 2."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1c, answer1c, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, identifying liquid Y and precipitate G, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the identification of liquid Y and precipitate G and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining identification of liquid Y and precipitate G, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = (
        "Sub-task 4: Analyze the product formed from the reaction of Substance X with a certain keto acid, "
        "focusing on the product containing 2 oxygen atoms, to confirm or refine the identification of Substance X and its isotopic composition, "
        "using output from Sub-task 1c."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1c, answer1c], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing keto acid product, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = (
        "Sub-task 5: Confirm the chemical nature and isotopic composition of Substance X and its close analog used as a reagent in organic chemistry, "
        "integrating all previous identifications, reaction characteristics, and isotopic data, using outputs from Sub-task 1c and Sub-task 4."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking1c, answer1c, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking1c, answer1c, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, confirming chemical nature and isotopic composition, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on chemical nature and isotopic composition of Substance X.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding chemical nature and isotopic composition, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6 = (
        "Sub-task 6: List all elements present in Substance X, including all isotopes (heavier and lighter), with their exact isotope masses and counts, "
        "and identify the lightest and heaviest elements among them, based on Sub-task 5 output."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, listing elements and isotopes, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7 = (
        "Sub-task 7: Calculate the cumulative atomic masses of the lightest and heaviest elements present in Substance X by summing the masses of all their isotopes multiplied by their counts, "
        "using precise isotope masses (e.g., C-12 = 12.000, C-13 = 13.003, O-16 = 15.995, O-18 = 17.999, H-1 = 1.0078, D = 2.014), based on Sub-task 6 output."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, calculating cumulative atomic masses, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_sc_instruction_8 = (
        "Sub-task 8: Compare the calculated cumulative atomic masses with the given multiple-choice options (29, 25, 35, 31) and select the correct answer choice; "
        "if no exact match is found, provide a reasoned explanation and reconsider earlier assumptions if necessary, based on Sub-task 7 output."
    )
    cot_agents_8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_8 = []
    thinkingmapping_8 = {}
    answermapping_8 = {}
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_sc_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking8, answer8 = await cot_agents_8[i]([taskInfo, thinking7, answer7], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_8[i].id}, selecting correct answer choice, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers_8.append(answer8.content)
        thinkingmapping_8[answer8.content] = thinking8
        answermapping_8[answer8.content] = answer8
    answer8_content = Counter(possible_answers_8).most_common(1)[0][0]
    thinking8 = thinkingmapping_8[answer8_content]
    answer8 = answermapping_8[answer8_content]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
