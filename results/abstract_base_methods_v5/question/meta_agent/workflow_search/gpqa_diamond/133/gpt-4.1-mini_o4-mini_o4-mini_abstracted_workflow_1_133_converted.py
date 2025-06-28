async def forward_133(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    
    # Stage 1: Calculate moles of each reactant
    # Sub-task 1: Calculate moles of HCl
    cot_instruction_1 = "Sub-task 1: Calculate the moles of HCl present using its volume and molarity from the given data in taskInfo."
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculating moles of HCl, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Calculate moles of H2SO4 with Self-Consistency CoT
    cot_sc_instruction_2 = "Sub-task 2: Calculate the moles of H2SO4 present using its volume and molarity from the given data in taskInfo."
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents[i]([taskInfo], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, calculating moles of H2SO4, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    # Sub-task 3: Calculate moles of Ba(OH)2
    cot_instruction_3 = "Sub-task 3: Calculate the moles of Ba(OH)2 present using its volume and molarity from the given data in taskInfo."
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculating moles of Ba(OH)2, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 2: Calculate total moles of H+ and OH- ions
    # Sub-task 4: Calculate total moles of H+ ions from HCl and H2SO4
    cot_instruction_4 = "Sub-task 4: Determine total moles of H+ ions contributed by HCl and H2SO4 combined, considering H2SO4 dissociates into 2 H+ ions per molecule."
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking4, answer4 = await cot_agent(cot_inputs_4, cot_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, calculating total moles of H+, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], "Review the total moles of H+ calculation for correctness and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback on total moles of H+, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent(cot_inputs_4, cot_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining total moles of H+, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Calculate total moles of OH- ions from Ba(OH)2
    cot_instruction_5 = "Sub-task 5: Determine total moles of OH- ions contributed by Ba(OH)2, considering it dissociates into 2 OH- ions per molecule."
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculating total moles of OH-, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    # Stage 3: Identify limiting reactant and calculate moles of water formed
    # Sub-task 6: Identify limiting reactant by comparing total moles of H+ and OH-
    cot_instruction_6 = "Sub-task 6: Identify the limiting reactant (H+ or OH-) by comparing total moles of H+ and OH- ions to find which ion is completely neutralized."
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking4, answer4, thinking5, answer5],
        "agent_collaboration": "Debate"
    }
    all_thinking6 = [[] for _ in range(self.max_round)]
    all_answer6 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, cot_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, identifying limiting reactant, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    thinking6, answer6 = await final_decision_agent([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on limiting reactant.", is_sub_task=True)
    agents.append(f"Final Decision agent, identifying limiting reactant, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    # Sub-task 7: Calculate moles of water formed equal to limiting reactant moles
    cot_instruction_7 = "Sub-task 7: Calculate the moles of water formed, which equals the moles of the limiting reactant (either H+ or OH-)."
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6, answer6],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculating moles of water formed, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    # Stage 4: Calculate enthalpy change and convert units
    # Sub-task 8: Calculate total enthalpy change using standard enthalpy per mole water formed
    cot_instruction_8 = "Sub-task 8: Use the standard enthalpy of neutralization (-57.3 kJ/mol) to calculate total enthalpy change for the neutralization reaction."
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking7, answer7],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_8 = [taskInfo, thinking7, answer7]
    thinking8, answer8 = await cot_agent(cot_inputs_8, cot_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, calculating enthalpy change, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking8, answer8], "Review enthalpy change calculation for accuracy and unit correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback on enthalpy change, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent(cot_inputs_8, cot_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining enthalpy change, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    # Sub-task 9: Convert enthalpy change to units used in answer choices
    cot_instruction_9 = "Sub-task 9: Convert the calculated enthalpy change into the units used in the answer choices (kcal or kJ) as needed."
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", thinking8, answer8],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent([taskInfo, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, converting enthalpy units, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    # Stage 5: Select correct answer choice
    # Sub-task 10: Compare calculated enthalpy with given options and select correct answer
    debate_instruction_10 = "Sub-task 10: Compare the calculated enthalpy of neutralization with the given multiple-choice options and select the correct answer choice (A, B, C, or D)."
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": debate_instruction_10,
        "context": ["user query", thinking9, answer9],
        "agent_collaboration": "Debate"
    }
    all_thinking10 = [[] for _ in range(self.max_round)]
    all_answer10 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking10, answer10 = await agent([taskInfo, thinking9, answer9], debate_instruction_10, r, is_sub_task=True)
            else:
                input_infos_10 = [taskInfo, thinking9, answer9] + all_thinking10[r-1] + all_answer10[r-1]
                thinking10, answer10 = await agent(input_infos_10, debate_instruction_10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct answer, thinking: {thinking10.content}; answer: {answer10.content}")
            all_thinking10[r].append(thinking10)
            all_answer10[r].append(answer10)
    thinking10, answer10 = await final_decision_agent([taskInfo] + all_thinking10[-1] + all_answer10[-1], "Sub-task 10: Make final decision on correct answer choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct answer, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
