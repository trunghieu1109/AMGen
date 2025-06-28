async def forward_71(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_agents_num = self.max_sc
    reflexion_max_round = self.max_round
    debate_rounds = self.max_round
    
    cot_instruction_1 = (
        "Sub-task 1: Establish a consistent atom numbering scheme and provide a fully detailed, stereochemically explicit, atom-numbered structural representation "
        "(including connectivity and stereochemistry) for the starting material 7-(tert-butoxy)bicyclo[2.2.1]hepta-2,5-diene. "
        "Include explicit atom numbering for all carbons, oxygens, and relevant substituents to serve as a reference for all subsequent transformations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, establishing atom numbering and detailed structure of starting material, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2 = (
        "Sub-task 2: Generate a fully atom-numbered, stereochemically explicit structure of the reagent 5,6-bis(dibromomethyl)cyclohexa-1,3-diene and sodium iodide. "
        "Clarify reactive sites and possible mechanistic roles in the reaction with the starting material. Use the atom numbering scheme consistent with Sub-task 1 where applicable."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_agents_num)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_agents_num):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, generating atom-numbered reagent structure and reactive sites, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_instruction_3 = (
        "Sub-task 3: Elucidate the detailed reaction mechanism between the starting material and 2 equivalents of 5,6-bis(dibromomethyl)cyclohexa-1,3-diene with sodium iodide at elevated temperature, producing product 1. "
        "Provide a fully atom-numbered, stereochemically explicit structure of product 1 with all connectivity changes clearly annotated, referencing the established atom numbering."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_agents_num)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_agents_num):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, elucidating reaction mechanism and product 1 structure, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = (
        "Sub-task 4: Analyze the reaction of product 1 with aqueous sulfuric acid to form product 2. "
        "Detail the mechanistic steps, protonation sites, and structural modifications, and provide a fully atom-numbered, stereochemically explicit structure of product 2. "
        "Ensure consistency with previous atom numbering."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, analyzing product 1 to product 2 transformation, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(reflexion_max_round):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the mechanistic steps, protonation sites, and structural modifications in product 2 formation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining product 2 analysis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5 = (
        "Sub-task 5: Analyze the reaction of product 2 with SO3 and pyridine in DMSO to form product 3. "
        "Include detailed mechanistic reasoning for functional group transformations and provide a fully atom-numbered, stereochemically explicit structure of product 3, consistent with previous numbering."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, analyzing product 2 to product 3 transformation, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(reflexion_max_round):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please review the functional group transformations and structural changes in product 3 formation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining product 3 analysis, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6 = (
        "Sub-task 6: Elucidate the thermal transformation of product 3 upon heating at 150°C to form the final product 4. "
        "Include mechanistic steps and structural rearrangements, providing a fully atom-numbered, stereochemically explicit structure of product 4 consistent with previous numbering."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, analyzing thermal treatment to final product 4, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(reflexion_max_round):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please review the reaction mechanism and structural features of final product 4 and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final product 4 analysis, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    debate_instruction_7 = (
        "Sub-task 7: Verify the final product 4's structure for chemical consistency, including mass balance, stereochemistry, and connectivity, "
        "ensuring it aligns with all previous transformations and atom numbering. Provide detailed justification referencing atom numbering and connectivity."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = [[] for _ in range(debate_rounds)]
    all_answer7 = [[] for _ in range(debate_rounds)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    for r in range(debate_rounds):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying final product 4 structure, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the chemical consistency and correctness of final product 4 structure.", is_sub_task=True)
    agents.append(f"Final Decision agent on final product 4 structure, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = (
        "Sub-task 8: Enumerate and classify all hydrogen atoms in the final product 4 according to their chemical environments (e.g., axial/equatorial, bridgehead, substituent positions), "
        "using the established atom numbering and structural details. Identify chemically distinct hydrogens based on symmetry and environment, providing detailed justification."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_8 = [taskInfo, thinking7, answer7]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, enumerating and classifying chemically distinct hydrogens, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(reflexion_max_round):
        feedback, correct = await critic_agent_8([taskInfo, thinking8, answer8], "please review the enumeration and classification of chemically distinct hydrogens and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining hydrogen enumeration, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    debate_instruction_9 = (
        "Sub-task 9: Compare the number of chemically distinct hydrogen atoms identified in Sub-task 8 with the provided multiple-choice options (7, 8, 10, 4) and select the correct answer choice. "
        "Provide justification based on structural analysis and atom numbering."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking9 = [[] for _ in range(debate_rounds)]
    all_answer9 = [[] for _ in range(debate_rounds)]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": debate_instruction_9,
        "context": ["user query", "thinking of subtask 8", "answer of subtask 8"],
        "agent_collaboration": "Debate"
    }
    for r in range(debate_rounds):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking8, answer8], debate_instruction_9, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct hydrogen count answer, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the correct number of chemically distinct hydrogens.", is_sub_task=True)
    agents.append(f"Final Decision agent on correct hydrogen count, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
