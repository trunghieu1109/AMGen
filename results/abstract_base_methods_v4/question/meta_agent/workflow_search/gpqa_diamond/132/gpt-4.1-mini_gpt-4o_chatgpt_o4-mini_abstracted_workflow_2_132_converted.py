async def forward_132(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Calculate the number of moles of KH2PO4 and Na2HPO4·2H2O in the solution using their given masses and molecular weights."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating moles of KH2PO4 and Na2HPO4·2H2O, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction_2 = "Sub-task 2: Convert the total volume of the solution from 200.00 cm³ to liters for concentration calculations."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, converting volume from cm³ to liters, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction_3 = "Sub-task 3: Calculate the molar concentrations of KH2PO4 and Na2HPO4·2H2O in the solution by dividing their moles from Sub-task 1 by the solution volume in liters from Sub-task 2."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculating molar concentrations, thinking: {thinking3.content}; answer: {answer3.content}")
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
    cot_instruction_4 = "Sub-task 4: Identify the phosphate species contributed by KH2PO4 and Na2HPO4·2H2O and summarize the relevant dissociation equilibria using the given Ka1, Ka2, and Ka3 values for H3PO4."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, identifying phosphate species and equilibria, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Set up the full system of equilibrium expressions for the phosphate buffer system at 25 °C, explicitly including mass balance, charge balance, and the dissociation equilibria of H3PO4 (Ka1, Ka2, Ka3), using initial concentrations from Sub-task 3 and species from Sub-task 4."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, setting up equilibrium expressions, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on equilibrium expressions setup.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding equilibrium setup, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_reflect_instruction_6_1 = "Sub-task 6.1: Calculate the pH of the solution by solving the nonlinear equilibrium system that includes all relevant phosphate species, mass balance, and charge balance equations without oversimplified assumptions, using the equilibrium setup from Sub-task 5."
    cot_agent_6_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6_1 = self.max_round
    cot_inputs_6_1 = [taskInfo, thinking5, answer5]
    subtask_desc6_1 = {
        "subtask_id": "subtask_6_1",
        "instruction": cot_reflect_instruction_6_1,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    thinking6_1, answer6_1 = await cot_agent_6_1(cot_inputs_6_1, cot_reflect_instruction_6_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6_1.id}, calculating pH by solving nonlinear system, thinking: {thinking6_1.content}; answer: {answer6_1.content}")
    for i in range(N_max_6_1):
        feedback6_1, correct6_1 = await critic_agent_6_1([taskInfo, thinking6_1, answer6_1], "Review the pH calculation and nonlinear equilibrium solution for completeness and accuracy.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6_1.id}, feedback on pH calculation, thinking: {feedback6_1.content}; answer: {correct6_1.content}")
        if correct6_1.content == "True":
            break
        cot_inputs_6_1.extend([thinking6_1, answer6_1, feedback6_1])
        thinking6_1, answer6_1 = await cot_agent_6_1(cot_inputs_6_1, cot_reflect_instruction_6_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6_1.id}, refining pH calculation, thinking: {thinking6_1.content}; answer: {answer6_1.content}")
    sub_tasks.append(f"Sub-task 6.1 output: thinking - {thinking6_1.content}; answer - {answer6_1.content}")
    subtask_desc6_1['response'] = {
        "thinking": thinking6_1,
        "answer": answer6_1
    }
    logs.append(subtask_desc6_1)
    print("Step 6.1: ", sub_tasks[-1])
    cot_reflect_instruction_6_2 = "Sub-task 6.2: Using the calculated pH from Sub-task 6.1, determine the distribution of phosphate species and specifically calculate the concentration of orthophosphate ions (PO4^3-) in the solution."
    cot_agent_6_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6_2 = self.max_round
    cot_inputs_6_2 = [taskInfo, thinking6_1, answer6_1]
    subtask_desc6_2 = {
        "subtask_id": "subtask_6_2",
        "instruction": cot_reflect_instruction_6_2,
        "context": ["user query", "thinking of subtask 6_1", "answer of subtask 6_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking6_2, answer6_2 = await cot_agent_6_2(cot_inputs_6_2, cot_reflect_instruction_6_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6_2.id}, calculating phosphate species distribution and [PO4^3-] concentration, thinking: {thinking6_2.content}; answer: {answer6_2.content}")
    for i in range(N_max_6_2):
        feedback6_2, correct6_2 = await critic_agent_6_2([taskInfo, thinking6_2, answer6_2], "Review the phosphate species distribution and orthophosphate ion concentration calculation for accuracy and assumptions.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6_2.id}, feedback on species distribution and [PO4^3-] calculation, thinking: {feedback6_2.content}; answer: {correct6_2.content}")
        if correct6_2.content == "True":
            break
        cot_inputs_6_2.extend([thinking6_2, answer6_2, feedback6_2])
        thinking6_2, answer6_2 = await cot_agent_6_2(cot_inputs_6_2, cot_reflect_instruction_6_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6_2.id}, refining phosphate species distribution and [PO4^3-] concentration, thinking: {thinking6_2.content}; answer: {answer6_2.content}")
    sub_tasks.append(f"Sub-task 6.2 output: thinking - {thinking6_2.content}; answer - {answer6_2.content}")
    subtask_desc6_2['response'] = {
        "thinking": thinking6_2,
        "answer": answer6_2
    }
    logs.append(subtask_desc6_2)
    print("Step 6.2: ", sub_tasks[-1])
    cot_instruction_6_3 = "Sub-task 6.3: Perform an independent arithmetic verification of the calculated orthophosphate ion concentration from Sub-task 6.2 to ensure numerical accuracy and consistency."
    cot_agent_6_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6_3 = {
        "subtask_id": "subtask_6_3",
        "instruction": cot_instruction_6_3,
        "context": ["user query", "thinking of subtask 6_2", "answer of subtask 6_2"],
        "agent_collaboration": "CoT"
    }
    thinking6_3, answer6_3 = await cot_agent_6_3([taskInfo, thinking6_2, answer6_2], cot_instruction_6_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6_3.id}, verifying arithmetic of [PO4^3-] concentration, thinking: {thinking6_3.content}; answer: {answer6_3.content}")
    sub_tasks.append(f"Sub-task 6.3 output: thinking - {thinking6_3.content}; answer - {answer6_3.content}")
    subtask_desc6_3['response'] = {
        "thinking": thinking6_3,
        "answer": answer6_3
    }
    logs.append(subtask_desc6_3)
    print("Step 6.3: ", sub_tasks[-1])
    cot_reflect_instruction_6_4 = "Sub-task 6.4: Validate the assumptions made during equilibrium calculations (e.g., negligible [H+]) and, if invalid, iteratively refine the calculations until self-consistency is achieved, using outputs from Sub-task 6.3."
    cot_agent_6_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6_4 = self.max_round
    cot_inputs_6_4 = [taskInfo, thinking6_3, answer6_3]
    subtask_desc6_4 = {
        "subtask_id": "subtask_6_4",
        "instruction": cot_reflect_instruction_6_4,
        "context": ["user query", "thinking of subtask 6_3", "answer of subtask 6_3"],
        "agent_collaboration": "Reflexion"
    }
    thinking6_4, answer6_4 = await cot_agent_6_4(cot_inputs_6_4, cot_reflect_instruction_6_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6_4.id}, validating assumptions and refining equilibrium calculations, thinking: {thinking6_4.content}; answer: {answer6_4.content}")
    for i in range(N_max_6_4):
        feedback6_4, correct6_4 = await critic_agent_6_4([taskInfo, thinking6_4, answer6_4], "Review the validity of assumptions and consistency of equilibrium calculations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6_4.id}, feedback on assumptions validation, thinking: {feedback6_4.content}; answer: {correct6_4.content}")
        if correct6_4.content == "True":
            break
        cot_inputs_6_4.extend([thinking6_4, answer6_4, feedback6_4])
        thinking6_4, answer6_4 = await cot_agent_6_4(cot_inputs_6_4, cot_reflect_instruction_6_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6_4.id}, refining assumptions validation, thinking: {thinking6_4.content}; answer: {answer6_4.content}")
    sub_tasks.append(f"Sub-task 6.4 output: thinking - {thinking6_4.content}; answer - {answer6_4.content}")
    subtask_desc6_4['response'] = {
        "thinking": thinking6_4,
        "answer": answer6_4
    }
    logs.append(subtask_desc6_4)
    print("Step 6.4: ", sub_tasks[-1])
    cot_instruction_7 = "Sub-task 7: Compare the verified orthophosphate ion concentration from Sub-task 6.4 with the provided multiple-choice options and select the correct answer."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6_4", "answer of subtask 6_4"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6_4, answer6_4], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, selecting correct multiple-choice answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs