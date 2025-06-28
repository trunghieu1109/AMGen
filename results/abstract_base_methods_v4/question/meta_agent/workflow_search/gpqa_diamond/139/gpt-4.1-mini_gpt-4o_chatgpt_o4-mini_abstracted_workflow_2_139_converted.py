async def forward_139(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1a = (
        "Subtask 1a: Enumerate all plausible classes of compounds that incorporate a heavier isotope "
        "of one of their constituent elements, considering the clues about Substance X such as heavier isotopes (e.g., deuterium), "
        "reaction with liquid Y, gas W, precipitate G, and organic chemistry usage."
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
        agents.append(f"CoT-SC agent {cot_agents_1a[i].id}, enumerating plausible compound classes, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a
    answer1a_content = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[answer1a_content]
    answer1a = answermapping_1a[answer1a_content]
    sub_tasks.append(f"Subtask 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_sc_instruction_1b = (
        "Subtask 1b: Generate specific candidate substances within each identified class from Subtask 1a, "
        "listing their chemical formulas and isotopic compositions including heavier isotopes such as deuterium."
    )
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1b, answer1b = await cot_agents_1b[i]([taskInfo, thinking1a, answer1a], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, generating candidate substances, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b
    answer1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[answer1b_content]
    answer1b = answermapping_1b[answer1b_content]
    sub_tasks.append(f"Subtask 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])

    cot_sc_instruction_1c = (
        "Subtask 1c: Systematically evaluate each candidate substance from Subtask 1b against all provided clues: "
        "presence of heavier isotope, reaction with liquid Y producing gas W, formation of precipitate G releasing B upon heating, "
        "melting point of B near 277 K, reaction with keto acid producing a product with 2 oxygen atoms, and usage in organic chemistry."
    )
    cot_agents_1c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_sc_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1c, answer1c = await cot_agents_1c[i]([taskInfo, thinking1b, answer1b], cot_sc_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, evaluating candidates against clues, thinking: {thinking1c.content}; answer: {answer1c.content}")
        possible_answers_1c.append(answer1c.content)
        thinkingmapping_1c[answer1c.content] = thinking1c
        answermapping_1c[answer1c.content] = answer1c
    answer1c_content = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking1c = thinkingmapping_1c[answer1c_content]
    answer1c = answermapping_1c[answer1c_content]
    sub_tasks.append(f"Subtask 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc1c['response'] = {"thinking": thinking1c, "answer": answer1c}
    logs.append(subtask_desc1c)
    print("Step 1c: ", sub_tasks[-1])

    cot_instruction_1d = (
        "Subtask 1d: Rank the candidate substances based on their consistency with all clues and select the top candidate(s) "
        "for further analysis."
    )
    cot_agent_1d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1d = {
        "subtask_id": "subtask_1d",
        "instruction": cot_instruction_1d,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "CoT"
    }
    thinking1d, answer1d = await cot_agent_1d([taskInfo, thinking1c, answer1c], cot_instruction_1d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1d.id}, ranking candidates and selecting top, thinking: {thinking1d.content}; answer: {answer1d.content}")
    sub_tasks.append(f"Subtask 1d output: thinking - {thinking1d.content}; answer - {answer1d.content}")
    subtask_desc1d['response'] = {"thinking": thinking1d, "answer": answer1d}
    logs.append(subtask_desc1d)
    print("Step 1d: ", sub_tasks[-1])

    cot_reflect_instruction_1e = (
        "Subtask 1e: Reflexively verify the top candidate(s) from Subtask 1d by cross-checking each clue for consistency; "
        "if inconsistencies are found, iterate back to Subtask 1b to reconsider candidates."
    )
    cot_agent_1e = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1e = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1e = self.max_round
    cot_inputs_1e = [taskInfo, thinking1d, answer1d]
    subtask_desc1e = {
        "subtask_id": "subtask_1e",
        "instruction": cot_reflect_instruction_1e,
        "context": ["user query", "thinking of subtask_1d", "answer of subtask_1d"],
        "agent_collaboration": "Reflexion"
    }
    thinking1e, answer1e = await cot_agent_1e(cot_inputs_1e, cot_reflect_instruction_1e, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1e.id}, verifying top candidate consistency, thinking: {thinking1e.content}; answer: {answer1e.content}")
    for i in range(N_max_1e):
        feedback, correct = await critic_agent_1e([taskInfo, thinking1e, answer1e], "Review top candidate consistency and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1e.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1e.extend([thinking1e, answer1e, feedback])
        thinking1e, answer1e = await cot_agent_1e(cot_inputs_1e, cot_reflect_instruction_1e, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1e.id}, refining verification, thinking: {thinking1e.content}; answer: {answer1e.content}")
    sub_tasks.append(f"Subtask 1e output: thinking - {thinking1e.content}; answer - {answer1e.content}")
    subtask_desc1e['response'] = {"thinking": thinking1e, "answer": answer1e}
    logs.append(subtask_desc1e)
    print("Step 1e: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Subtask 2: Identify the gas W produced in the reaction of Substance X with liquid Y, using the verified identity of Substance X "
        "from Subtask 1e and the clue that gas W's molecule contains equal numbers of neutrons and protons."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask_1e", "answer of subtask_1e"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1e, answer1e], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying gas W, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Subtask 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction_3 = (
        "Subtask 3: Identify the precipitate G formed in the reaction and analyze the nature of substance B released upon heating G, "
        "confirming B's identity by correlating with its melting point near 277 K."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1e, answer1e, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask_1e", "answer of subtask_1e", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, analyzing precipitate G and B, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "Review identification of precipitate G and B, provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining analysis of G and B, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Subtask 4: Analyze the reaction product of Substance X with a keto acid that contains 2 oxygen atoms, "
        "to confirm the chemical nature and composition of Substance X and its close analog used as a reagent in organic chemistry."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask_1e", "answer of subtask_1e"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1e, answer1e], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyzing keto acid reaction, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Subtask 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Subtask 5: List all elements present in the verified Substance X, including all isotopic variants (heavier and lighter isotopes), "
        "and identify the lightest and heaviest elements among them."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_1e", "answer of subtask_1e", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking1e, answer1e, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking1e, answer1e, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, listing elements and identifying extremes, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Subtask 5: Make final decision on elements present and their lightest and heaviest identities.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing elements and extremes, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_6 = (
        "Subtask 6: Calculate the cumulative atomic masses of the lightest and heaviest elements present in Substance X by summing the masses "
        "of all heavier and lighter isotopes for each element, considering multiple instances if present."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask_5", "answer of subtask_5"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating cumulative atomic masses, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Subtask 6: Make final decision on cumulative atomic masses.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing cumulative atomic masses, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_reflect_instruction_7 = (
        "Subtask 7: Validate the calculated cumulative atomic masses from Subtask 6 against the provided multiple-choice options (29, 25, 35, 31), "
        "and select the correct answer. If no exact match is found, perform reflexion to reconsider assumptions or suggest the best possible choice with justification."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", "thinking of subtask_6", "answer of subtask_6"],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, validating cumulative masses against options, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7], "Review validation of cumulative atomic masses and choice selection.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining validation and choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
