async def forward_91(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]

    cot_instruction_1a = (
        "Sub-task 1a: Systematically enumerate all atoms and substituents in the molecule (CH3)2C=CH-CH2-CH(CH3)-CH2-CH=C(CH3)2 by explicitly listing each fragment and its connectivity to accurately determine the total number of carbon and hydrogen atoms. "
        "Provide a clear structural enumeration including all substituents and branches."
    )
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}

    for i in range(self.max_sc):
        thinking1a, answer1a = await cot_sc_agents_1a[i]([taskInfo], cot_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1a[i].id}, enumerate atoms and substituents, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a

    answer1a_content = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[answer1a_content]
    answer1a = answermapping_1a[answer1a_content]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_instruction_1b = (
        "Sub-task 1b: Count and tabulate each bond type (C-C single bonds, C=C double bonds, C-H bonds) in the molecule based on the explicit enumeration from subtask 1a, "
        "ensuring all substituents and branches are included. Provide a clear table or list of bond counts."
    )
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}

    for i in range(self.max_sc):
        thinking1b, answer1b = await cot_sc_agents_1b[i]([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1b[i].id}, count bond types, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b

    answer1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[answer1b_content]
    answer1b = answermapping_1b[answer1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])

    cot_instruction_1c = (
        "Sub-task 1c: Calculate the degree of unsaturation (DU) using the formula DU = (2C + 2 - H)/2 to verify the consistency of atom counts and the number of double bonds identified, "
        "cross-checking results from subtasks 1a and 1b. Confirm if the DU matches the number of double bonds counted."
    )
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "answer of subtask_1a", "answer of subtask_1b"],
        "agent_collaboration": "CoT"
    }
    thinking1c, answer1c = await cot_agent_1c([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, calculate degree of unsaturation, thinking: {thinking1c.content}; answer: {answer1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking1c, "answer": answer1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])

    cot_instruction_1d = (
        "Sub-task 1d: Perform a self-consistency check by generating at least two independent parses of the molecular structure and bond counts, "
        "then compare and reconcile differences to finalize accurate atom and bond counts. Provide a consensus result."
    )
    cot_sc_agents_1d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    subtask_desc_1d = {
        "subtask_id": "subtask_1d",
        "instruction": cot_instruction_1d,
        "context": ["user query", "answer of subtask_1a", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_1d = []
    thinkingmapping_1d = {}
    answermapping_1d = {}

    for i in range(self.max_sc):
        thinking1d, answer1d = await cot_sc_agents_1d[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_instruction_1d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1d[i].id}, self-consistency check on atom and bond counts, thinking: {thinking1d.content}; answer: {answer1d.content}")
        possible_answers_1d.append(answer1d.content)
        thinkingmapping_1d[answer1d.content] = thinking1d
        answermapping_1d[answer1d.content] = answer1d

    answer1d_content = Counter(possible_answers_1d).most_common(1)[0][0]
    thinking1d = thinkingmapping_1d[answer1d_content]
    answer1d = answermapping_1d[answer1d_content]
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking1d.content}; answer - {answer1d.content}")
    subtask_desc_1d['response'] = {"thinking": thinking1d, "answer": answer1d}
    logs.append(subtask_desc_1d)
    print("Step 1d: ", sub_tasks[-1])

    cot_reflect_instruction_1e = (
        "Sub-task 1e: Reflect on and verify the finalized atom and bond counts for internal consistency and correctness before proceeding to energy calculations. "
        "Check for any discrepancies or inconsistencies and confirm final counts."
    )
    cot_agent_1e = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1e = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1e = self.max_round
    cot_inputs_1e = [taskInfo, thinking1c, answer1c, thinking1d, answer1d]
    subtask_desc_1e = {
        "subtask_id": "subtask_1e",
        "instruction": cot_reflect_instruction_1e,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c", "thinking of subtask_1d", "answer of subtask_1d"],
        "agent_collaboration": "Reflexion"
    }
    thinking1e, answer1e = await cot_agent_1e(cot_inputs_1e, cot_reflect_instruction_1e, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1e.id}, verifying atom and bond counts, thinking: {thinking1e.content}; answer: {answer1e.content}")

    for i in range(N_max_1e):
        feedback, correct = await critic_agent_1e([taskInfo, thinking1e, answer1e],
                                                 "Please review the finalized atom and bond counts for correctness and internal consistency.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1e.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1e.extend([thinking1e, answer1e, feedback])
        thinking1e, answer1e = await cot_agent_1e(cot_inputs_1e, cot_reflect_instruction_1e, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1e.id}, refining verification, thinking: {thinking1e.content}; answer: {answer1e.content}")

    sub_tasks.append(f"Sub-task 1e output: thinking - {thinking1e.content}; answer - {answer1e.content}")
    subtask_desc_1e['response'] = {"thinking": thinking1e, "answer": answer1e}
    logs.append(subtask_desc_1e)
    print("Step 1e: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Calculate the total bond energy of the molecule by multiplying the verified number of each bond type (C-C, C=C, C-H) by their respective bond energies "
        "(C-C = 200 kJ/mol, C=C = 300 kJ/mol, C-H = 400 kJ/mol) and summing these values. Use the verified bond counts from subtask 1e."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "answer of subtask_1e"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1e, answer1e], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculate total bond energy, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Calculate the total enthalpy of atomization of all carbon atoms by multiplying the verified number of carbon atoms by the enthalpy of atomization of carbon (1000 kJ/mol). "
        "Use the verified atom counts from subtask 1e."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "answer of subtask_1e"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1e, answer1e], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculate enthalpy of atomization of carbon atoms, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Calculate the total enthalpy of atomization of all hydrogen atoms by using the bond energy of H-H (100 kJ/mol) and the verified number of hydrogen atoms, "
        "considering that each H-H bond corresponds to two hydrogen atoms. Use the verified atom counts from subtask 1e."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "answer of subtask_1e"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1e, answer1e], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculate enthalpy of atomization of hydrogen atoms, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    cot_reflect_instruction_5 = (
        "Sub-task 5: Reflect on and verify the intermediate energy values (bond energies and atomization enthalpies) for thermochemical consistency and correctness before final enthalpy calculation. "
        "Check for any discrepancies or inconsistencies in values from subtasks 2, 3, and 4."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_3", "answer of subtask_3", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying intermediate energy values, thinking: {thinking5.content}; answer: {answer5.content}")

    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5],
                                                 "Please review the intermediate energy values for thermochemical consistency and correctness.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")

    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Calculate the enthalpy of formation of the molecule using the formula: Enthalpy of formation = (Sum of enthalpy of atomization of atoms) - (Sum of bond energies in the molecule), "
        "using verified values from subtasks 2, 3, and 4."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "answer of subtask_2", "answer of subtask_3", "answer of subtask_4", "answer of subtask_5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, answer2, answer3, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculate enthalpy of formation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc_6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Calculate the molecular weight of the molecule based on the verified atom counts to enable unit conversion between kJ/mol and kJ/g. "
        "Use the verified atom counts from subtask 1e."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "answer of subtask_1e"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking1e, answer1e], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, calculate molecular weight, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc_7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Convert the calculated enthalpy of formation from kJ/mol to kJ/g using the molecular weight from subtask 7, ensuring unit consistency with the multiple-choice options. "
        "Show the conversion steps clearly."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "answer of subtask_6", "answer of subtask_7"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, answer6, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, convert enthalpy units, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc_8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc_8)
    print("Step 8: ", sub_tasks[-1])

    debate_instruction_9 = (
        "Sub-task 9: Debate and reflect on the calculated enthalpy values (both kJ/mol and kJ/g) and compare them with the given multiple-choice options: "
        "A: 11200 kJ/mol, B: 1900 kJ/g, C: 11.44 kJ/g, D: 67.4 kJ/mol. Select the correct choice letter that best matches the calculation. "
        "Discuss unit consistency, numerical closeness, and reasoning behind the choice."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    subtask_desc_9 = {
        "subtask_id": "subtask_9",
        "instruction": debate_instruction_9,
        "context": ["user query", "answer of subtask_6", "answer of subtask_8"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, answer6, answer8], debate_instruction_9, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, answer6, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final choice, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)

    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1],
                                                    "Sub-task 9: Make final decision on the correct multiple-choice option (A, B, C, or D) based on debate outputs.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final choice, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc_9 = {
        "thinking": thinking9,
        "answer": answer9
    }
    logs.append({
        "subtask_id": "subtask_9",
        "instruction": debate_instruction_9,
        "context": ["user query", "answer of subtask_6", "answer of subtask_8"],
        "agent_collaboration": "Debate",
        "response": subtask_desc_9
    })
    print("Step 9: ", sub_tasks[-1])

    cot_instruction_10 = (
        "Sub-task 10: Validate the final answer format to ensure compliance with the instruction to return only the letter choice (A, B, C, or D). "
        "If no option matches, trigger re-examination of previous subtasks."
    )
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction_10,
        "context": ["user query", "answer of subtask_9"],
        "agent_collaboration": "CoT"
    }
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking9, answer9], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, validate final answer format, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc_10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc_10)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
