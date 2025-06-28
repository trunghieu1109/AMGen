async def forward_36(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    N = self.max_sc
    N_max = self.max_round
    
    cot_instruction_1 = (
        "Sub-task 1: Determine the detailed chemical structure of intermediate A formed from the reaction of propionaldehyde "
        "with 1,3-ethanedithiol (EDT) and BF3. Provide the IUPAC name, skeletal formula, and carbon numbering scheme explicitly. "
        "Use unambiguous structural representations such as SMILES or InChI if possible."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, determining structure of intermediate A, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Determine the detailed chemical structure of intermediate B formed by treating A with BuLi. "
        "Specify changes in functional groups and carbon framework with explicit structural representation, including IUPAC name, skeletal formula, and carbon numbering. "
        "Use outputs from Sub-task 1 as context."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining structure of intermediate B, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_sc_instruction_3 = (
        "Sub-task 3: Determine the detailed chemical structure of intermediate C formed by alkylation of B with bromoethane. "
        "Include updated IUPAC name, skeletal formula, and carbon numbering. Use outputs from Sub-task 2 as context."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, determining structure of intermediate C, thinking: {thinking3.content}; answer: {answer3.content}")
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
    
    cot_sc_instruction_4 = (
        "Sub-task 4: Apply HgCl2 / H2O / H+ treatment to intermediate C to obtain intermediate D. "
        "Use Self-Consistency Chain-of-Thought (SC-CoT) to generate multiple independent reasoning paths to confirm whether the product is a ketone or aldehyde. "
        "Provide IUPAC name, skeletal formula, oxidation state, and carbon numbering explicitly. Use outputs from Sub-task 3 as context."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, verifying product identity of intermediate D, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_reflect_instruction_5 = (
        "Sub-task 5: Perform a reflexion step to confirm and justify the oxidation state, functional group identity, and structure of intermediate D. "
        "Ensure consensus on the ketone nature and correct IUPAC naming before proceeding. Use outputs from Sub-task 4 as context."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, confirming intermediate D structure, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5],
                                                "Please review the confirmation of intermediate D's oxidation state and functional group identity, and provide limitations if any.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining confirmation of intermediate D, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    debate_instruction_6 = (
        "Sub-task 6: Determine the detailed chemical structure of the final product E formed by reacting D with PPh3, 3-bromopentane, and BuLi. "
        "Include explicit IUPAC name, skeletal formula, carbon numbering, and stereochemical considerations. Use outputs from Sub-task 5 as context. "
        "Use debate among agents to refine and finalize the structure."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = [[] for _ in range(N_max)]
    all_answer6 = [[] for _ in range(N_max)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining final product E structure, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the structure of final product E.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing product E structure, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7a = (
        "Sub-task 7a: Analyze the alkene carbons in final product E to identify unique carbon environments relevant for 13C-NMR spectroscopy. "
        "Consider molecular symmetry, chemical equivalence, stereochemistry, and conformational averaging. Use outputs from Sub-task 6 as context."
    )
    cot_agent_7a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7a = {
        "subtask_id": "subtask_7a",
        "instruction": cot_instruction_7a,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7a, answer7a = await cot_agent_7a([taskInfo, thinking6, answer6], cot_instruction_7a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7a.id}, analyzing alkene carbons in E, thinking: {thinking7a.content}; answer: {answer7a.content}")
    sub_tasks.append(f"Sub-task 7a output: thinking - {thinking7a.content}; answer - {answer7a.content}")
    subtask_desc7a['response'] = {"thinking": thinking7a, "answer": answer7a}
    logs.append(subtask_desc7a)
    print("Step 7a: ", sub_tasks[-1])
    
    cot_instruction_7b = (
        "Sub-task 7b: Analyze the ethyl substituent carbons in final product E to identify unique carbon environments for 13C-NMR. "
        "Consider symmetry, equivalence, stereochemistry, and conformational averaging. Use outputs from Sub-task 6 as context."
    )
    cot_agent_7b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7b = {
        "subtask_id": "subtask_7b",
        "instruction": cot_instruction_7b,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7b, answer7b = await cot_agent_7b([taskInfo, thinking6, answer6], cot_instruction_7b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7b.id}, analyzing ethyl substituent carbons in E, thinking: {thinking7b.content}; answer: {answer7b.content}")
    sub_tasks.append(f"Sub-task 7b output: thinking - {thinking7b.content}; answer - {answer7b.content}")
    subtask_desc7b['response'] = {"thinking": thinking7b, "answer": answer7b}
    logs.append(subtask_desc7b)
    print("Step 7b: ", sub_tasks[-1])
    
    cot_instruction_7c = (
        "Sub-task 7c: Analyze the pentyl substituent carbons in final product E to identify unique carbon environments for 13C-NMR. "
        "Explicitly consider molecular symmetry, chemical equivalence, stereochemistry, and conformational averaging. Use outputs from Sub-task 6 as context."
    )
    cot_agent_7c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7c = {
        "subtask_id": "subtask_7c",
        "instruction": cot_instruction_7c,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7c, answer7c = await cot_agent_7c([taskInfo, thinking6, answer6], cot_instruction_7c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7c.id}, analyzing pentyl substituent carbons in E, thinking: {thinking7c.content}; answer: {answer7c.content}")
    sub_tasks.append(f"Sub-task 7c output: thinking - {thinking7c.content}; answer - {answer7c.content}")
    subtask_desc7c['response'] = {"thinking": thinking7c, "answer": answer7c}
    logs.append(subtask_desc7c)
    print("Step 7c: ", sub_tasks[-1])
    
    cot_sc_instruction_7d = (
        "Sub-task 7d: Integrate the analyses from Sub-tasks 7a, 7b, and 7c to enumerate all unique carbon environments in final product E for 13C-NMR spectroscopy. "
        "Use Self-Consistency Chain-of-Thought (SC-CoT) to generate multiple independent assessments, explicitly justifying symmetry, chemical equivalence, and stereochemical considerations."
    )
    cot_agents_7d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_7d = []
    thinkingmapping_7d = {}
    answermapping_7d = {}
    subtask_desc7d = {
        "subtask_id": "subtask_7d",
        "instruction": cot_sc_instruction_7d,
        "context": ["user query", "thinking of subtask 7a", "answer of subtask 7a", "thinking of subtask 7b", "answer of subtask 7b", "thinking of subtask 7c", "answer of subtask 7c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking7d, answer7d = await cot_agents_7d[i]([
            taskInfo, thinking7a, answer7a, thinking7b, answer7b, thinking7c, answer7c
        ], cot_sc_instruction_7d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7d[i].id}, integrating carbon environment analyses, thinking: {thinking7d.content}; answer: {answer7d.content}")
        possible_answers_7d.append(answer7d.content)
        thinkingmapping_7d[answer7d.content] = thinking7d
        answermapping_7d[answer7d.content] = answer7d
    answer7d_content = Counter(possible_answers_7d).most_common(1)[0][0]
    thinking7d = thinkingmapping_7d[answer7d_content]
    answer7d = answermapping_7d[answer7d_content]
    sub_tasks.append(f"Sub-task 7d output: thinking - {thinking7d.content}; answer - {answer7d.content}")
    subtask_desc7d['response'] = {"thinking": thinking7d, "answer": answer7d}
    logs.append(subtask_desc7d)
    print("Step 7d: ", sub_tasks[-1])
    
    cot_reflect_instruction_8 = (
        "Sub-task 8: Perform a reflexion step on the initial 13C-NMR signal count from Sub-task 7d to detect and correct any overcounting or overlooked equivalences. "
        "Justify the final count with detailed reasoning and ensure consistency with molecular symmetry and stereochemistry."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_8 = [taskInfo, thinking7d, answer7d]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", "thinking of subtask 7d", "answer of subtask 7d"],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining 13C-NMR signal count, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_8([taskInfo, thinking8, answer8],
                                                "Critically evaluate the 13C-NMR signal count and provide limitations or corrections if any.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining 13C-NMR signal count, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    debate_instruction_9 = (
        "Sub-task 9: Compare the final verified number of distinct 13C-NMR signals with the provided multiple-choice options (3, 11, 8, 6) and select the correct choice. "
        "Provide a brief justification for the selection using debate among agents."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking9 = [[] for _ in range(N_max)]
    all_answer9 = [[] for _ in range(N_max)]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": debate_instruction_9,
        "context": ["user query", "thinking of subtask 8", "answer of subtask 8"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking8, answer8], debate_instruction_9, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct 13C-NMR signal count, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the correct number of 13C-NMR signals.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing 13C-NMR signal count, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
