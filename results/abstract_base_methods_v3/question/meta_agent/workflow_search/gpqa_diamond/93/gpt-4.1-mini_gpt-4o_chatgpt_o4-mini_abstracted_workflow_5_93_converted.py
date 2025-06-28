async def forward_93(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the target molecule methyl 2-propyl-1,2,4a,5,6,7,8,8a-octahydronaphthalene-1-carboxylate to identify its core structural features, including the decalin (octahydronaphthalene) ring system, the position and nature of substituents (methyl ester at C-1, propyl group at C-2), and saturation level."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing target molecule features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    cot_instruction_1_5 = "Sub-task 1.5: Map each substituent of the target molecule, especially the propyl group at C-2 and the methyl ester at C-1, to potential fragments or moieties that must be present in the starting materials, ensuring explicit tracking of substituent origin for retrosynthetic analysis. Use structural representations or fragment mapping if possible."
    cot_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1_5 = {
        "subtask_id": "subtask_1.5",
        "instruction": cot_instruction_1_5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking1_5, answer1_5 = await cot_agent_1_5([taskInfo, thinking1, answer1], cot_instruction_1_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_5.id}, mapping substituents to fragments, thinking: {thinking1_5.content}; answer: {answer1_5.content}")
    sub_tasks.append(f"Sub-task 1.5 output: thinking - {thinking1_5.content}; answer - {answer1_5.content}")
    subtask_desc1_5['response'] = {
        "thinking": thinking1_5,
        "answer": answer1_5
    }
    logs.append(subtask_desc1_5)
    print("Subtask 1.5 answer: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Identify the key retrosynthetic disconnections and reaction types (notably Diels-Alder cycloaddition) required to construct the target molecule’s core and substituents, specifying the need for a conjugated diene and a suitable dienophile, and assign which substituent each fragment should supply, based on the mapping from Sub-task 1.5."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1.5", "answer of subtask 1.5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1_5, answer1_5], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying retrosynthetic disconnections, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])
    cot_sc_instruction_3a = "Sub-task 3a: Evaluate the diene candidates in each starting material choice for conjugation, ability to supply the decalin core, and the propyl substituent at C-2, based on retrosynthetic logic and chemical rules for Diels-Alder reactions."
    cot_sc_instruction_3b = "Sub-task 3b: Evaluate the dienophile candidates in each starting material choice for compatibility, ability to supply the methyl ester substituent at C-1, and chemical feasibility."
    cot_sc_instruction_3c = "Sub-task 3c: Combine the assessments from 3a and 3b to select the best diene-dienophile pair for each starting material choice, mapping substituents explicitly and justifying chemical feasibility."
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_3c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, evaluating diene candidates, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    most_common_answer_3a = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[most_common_answer_3a]
    answer3a = answermapping_3a[most_common_answer_3a]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Subtask 3a answer: ", sub_tasks[-1])
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, evaluating dienophile candidates, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    most_common_answer_3b = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[most_common_answer_3b]
    answer3b = answermapping_3b[most_common_answer_3b]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Subtask 3b answer: ", sub_tasks[-1])
    possible_answers_3c = []
    thinkingmapping_3c = {}
    answermapping_3c = {}
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_sc_instruction_3c,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3c, answer3c = await cot_agents_3c[i]([taskInfo, thinking3a, answer3a, thinking3b, answer3b], cot_sc_instruction_3c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3c[i].id}, combining diene and dienophile assessments, thinking: {thinking3c.content}; answer: {answer3c.content}")
        possible_answers_3c.append(answer3c.content)
        thinkingmapping_3c[answer3c.content] = thinking3c
        answermapping_3c[answer3c.content] = answer3c
    most_common_answer_3c = Counter(possible_answers_3c).most_common(1)[0][0]
    thinking3c = thinkingmapping_3c[most_common_answer_3c]
    answer3c = answermapping_3c[most_common_answer_3c]
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Subtask 3c answer: ", sub_tasks[-1])
    debate_instruction_4 = "Sub-task 4: Integrate the assessments from Sub-task 3 to compare all starting material choices, explicitly verifying that each choice’s starting materials can supply all required substituents at correct positions and satisfy chemical rules for Diels-Alder reactions (e.g., conjugated diene requirement), and eliminate unsuitable options. Challenge assumptions and ensure chemical feasibility."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3c, answer3c]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
                input_infos_4.extend(all_answer4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating assessments, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make a final decision on the most suitable starting material choice after validation.", is_sub_task=True)
    agents.append(f"Final Decision agent on starting material choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Subtask 4 answer: ", sub_tasks[-1])
    cot_reflect_instruction_5 = "Sub-task 5: Conduct a reflexive review of the reasoning and mapping from Sub-task 4 to confirm the selected starting materials choice is chemically sound, supplies all substituents correctly, and is consistent with retrosynthetic logic; if inconsistencies are found, suggest re-evaluation or corrections."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, reviewing starting material choice, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Please review the selected starting materials choice and provide any limitations or corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining review, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Subtask 5 answer: ", sub_tasks[-1])
    cot_instruction_6 = "Sub-task 6: Select and output the correct multiple-choice answer (A, B, C, or D) corresponding to the starting materials that can feasibly synthesize methyl 2-propyl-1,2,4a,5,6,7,8,8a-octahydronaphthalene-1-carboxylate based on validated retrosynthetic analysis from Sub-task 5."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Subtask 6 answer: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs