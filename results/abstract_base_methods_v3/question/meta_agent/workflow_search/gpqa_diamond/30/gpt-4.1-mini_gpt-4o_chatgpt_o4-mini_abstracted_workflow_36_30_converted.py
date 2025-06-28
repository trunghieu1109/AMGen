async def forward_30(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Determine the chemical structure of product 1 formed when toluene is treated with nitric acid and sulfuric acid, using knowledge of electrophilic aromatic substitution to identify the position of nitration (expected para-nitrotoluene)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, determining product 1 structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Determine the chemical structure of product 2 formed when product 1 is treated with MnO2 and H2SO4. Note: Under mild conditions, MnO2 oxidizes benzylic methyl groups primarily to aldehydes (para-nitrobenzaldehyde), not carboxylic acids. Include a brief reagent-function note and perform a self-consistency check to confirm the aldehyde structure before proceeding."
    N = self.max_sc
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining product 2 structure, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction_3a = "Sub-task 3a: Enumerate and describe all plausible reaction mechanisms between product 2 (para-nitrobenzaldehyde) and acetone in aqueous sodium hydroxide, focusing on base-catalyzed aldol condensation and other possible pathways."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, enumerating plausible reaction mechanisms, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    debate_instruction_3b = "Sub-task 3b: Debate the plausible reaction mechanisms from subtask 3a using chemical principles and reagent-specific knowledge to predict the most likely product(s), emphasizing the formation of chalcone-type α,β-unsaturated ketones via aldol condensation."
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3b = self.max_round
    all_thinking3b = [[] for _ in range(N_max_3b)]
    all_answer3b = [[] for _ in range(N_max_3b)]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": debate_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking3b, answer3b = await agent([taskInfo, thinking3a, answer3a], debate_instruction_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking3a, answer3a] + all_thinking3b[r-1] + all_answer3b[r-1]
                thinking3b, answer3b = await agent(input_infos_3b, debate_instruction_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating reaction mechanisms, thinking: {thinking3b.content}; answer: {answer3b.content}")
            all_thinking3b[r].append(thinking3b)
            all_answer3b[r].append(answer3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3b, answer3b = await final_decision_agent_3b([taskInfo] + all_thinking3b[-1] + all_answer3b[-1], "Sub-task 3b: Make final decision on the most likely product(s) from the debated reaction mechanisms.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding most likely product(s), thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    cot_reflect_instruction_3c = "Sub-task 3c: Confirm and clearly define the chemical structure of product 3 based on the selected reaction pathway from subtask 3b, ensuring the structure is chemically plausible and consistent with known reaction outcomes."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3c = self.max_round
    cot_inputs_3c = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3b, answer3b]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, confirming product 3 structure, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max_3c):
        feedback, correct = await critic_agent_3c([taskInfo, thinking3c, answer3c], "please review the chemical structure determination of product 3 and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining product 3 structure, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    cot_instruction_4 = "Sub-task 4: Analyze the confirmed chemical structure of product 3 to identify all molecular symmetry elements (mirror planes, rotation axes, inversion centers, etc.) relevant to molecular symmetry groups."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3c, answer3c], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing symmetry elements of product 3, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Classify the molecular symmetry group of product 3 based on the identified symmetry elements and match it to one of the given multiple-choice options: Cs, C2h, C3, or D2h."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
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
            agents.append(f"Debate agent {agent.id}, round {r}, classifying molecular symmetry group, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the molecular symmetry group classification.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining molecular symmetry group, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs