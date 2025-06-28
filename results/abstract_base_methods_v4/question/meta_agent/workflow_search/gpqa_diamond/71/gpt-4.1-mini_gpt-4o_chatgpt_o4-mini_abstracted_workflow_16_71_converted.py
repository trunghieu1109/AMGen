async def forward_71(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_agents_num = self.max_sc
    max_reflexion_rounds = self.max_round
    debate_rounds = self.max_round
    debate_roles = self.debate_role
    cot_temperature = 0.0
    cot_sc_temperature = 0.5
    debate_temperature = 0.5
    cot_agent_base = LLMAgentBase

    cot_instruction_1 = (
        "Sub-task 1: Explicitly determine and represent the chemical structure of product 1 formed by the reaction of 7-(tert-butoxy)bicyclo[2.2.1]hepta-2,5-diene with 2 equivalents of 5,6-bis(dibromomethyl)cyclohexa-1,3-diene and sodium iodide at elevated temperature. "
        "Identify the reaction mechanism(s), propose plausible product structures with detailed chemical reasoning, and provide machine-readable representations (SMILES or InChI) for product 1."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_1 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_sc_temperature) for _ in range(cot_sc_agents_num)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    for i in range(cot_sc_agents_num):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, determining product 1 structure, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    answer1_content = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[answer1_content]
    answer1 = answermapping_1[answer1_content]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Analyze the reaction of product 1 with aqueous sulfuric acid to determine the exact chemical transformation(s) occurring, "
        "deduce the structure of product 2 with explicit structural representation, and justify the changes based on known reaction mechanisms (e.g., hydrolysis, rearrangement). "
        "Provide updated machine-readable chemical structures for product 2."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_2 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_sc_temperature) for _ in range(cot_sc_agents_num)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(cot_sc_agents_num):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing product 1 reaction with aqueous sulfuric acid, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Determine the chemical transformation of product 2 upon treatment with SO3 and pyridine in DMSO, including the reaction mechanism (e.g., sulfonation), "
        "and deduce the explicit structure of product 3. Provide detailed structural representations and chemical reasoning supporting the proposed structure."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_3 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    critic_agent_3 = cot_agent_base(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining product 3 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(max_reflexion_rounds):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the deduced structure of product 3 and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining product 3 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Elucidate the structural changes when product 3 is heated at 150°C to form the final product 4, specifying the reaction type (e.g., elimination, rearrangement), "
        "and provide a fully detailed, validated chemical structure of product 4 with machine-readable format. Confirm chemical plausibility and consistency with prior intermediates."
    )
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_4 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature)
    critic_agent_4 = cot_agent_base(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, determining final product 4 structure, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(max_reflexion_rounds):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the deduced structure of final product 4 and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining final product 4 structure, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Perform a rigorous analysis of the final product 4's structure to identify and enumerate all chemically distinct hydrogen atoms. "
        "Assess symmetry, chemical environment, and stereochemistry to distinguish unique hydrogen environments. Provide detailed explanation and justification for each distinct hydrogen type identified."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    debate_agents_5 = [cot_agent_base(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=debate_temperature) for role in debate_roles]
    all_thinking5 = [[] for _ in range(debate_rounds)]
    all_answer5 = [[] for _ in range(debate_rounds)]
    for r in range(debate_rounds):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating distinct hydrogens, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the number of chemically distinct hydrogen atoms on product 4.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding number of distinct hydrogens, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Compare the enumerated number of chemically distinct hydrogen atoms on product 4 with the provided multiple-choice options (7, 8, 10, 4). "
        "Select and justify the correct answer choice based on the detailed structural analysis from subtask 5."
    )
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    cot_agent_6 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting correct multiple-choice answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
