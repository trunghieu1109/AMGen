async def forward_128(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the structure and functional groups of compound A by integrating the IR spectrum peak at ~1750 cm^-1 (indicative of a carbonyl group) and the reaction with an unknown phosphorous ylide that forms 1,2-dimethyl-4-(propan-2-ylidene)cyclopentane, to deduce the likely identity and structure of compound A."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing compound A structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Determine the structure of compound B formed by the nucleophilic addition of HCN to compound A, focusing on the formation of the cyanohydrin functional group and its stereochemical and structural implications, based on subtask 1 outputs."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining compound B structure, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_reflect_instruction_3 = "Sub-task 3: Identify the structure of compound C formed by catalytic hydrogenation (H2, Pd) of compound B, specifically the reduction of the nitrile group to a primary amine, and analyze the resulting functional groups and stereochemistry, based on outputs from subtasks 1 and 2."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, identifying compound C, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the identification of compound C and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback on compound C, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining compound C, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction_4 = "Sub-task 4: Determine the correct structure of compound D formed by the reaction of compound C with nitrous acid (HNO2), explicitly considering the formation of the diazonium salt intermediate (4-diazonio-4-hydroxy-1,2-dimethylcyclopentane). Conduct a multi-agent debate to evaluate alternative identities (diazonium salt vs. geminal diol vs. rearranged ketone), incorporating mechanistic reasoning, spectral features (e.g., possible N≡N stretch), and chemical stability to select the most plausible structure, based on outputs from subtasks 1-3."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking and answer of subtasks 1-3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating compound D identity, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Referee Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the identity of compound D, considering mechanistic plausibility, spectral data (e.g., N≡N stretch), and chemical stability.", is_sub_task=True)
    agents.append(f"Referee agent, deciding compound D, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction_5a = "Sub-task 5a: Enumerate all plausible elimination and rearrangement pathways from compound D to compound E, including ring expansion via semipinacol rearrangement and ring retention with methyl migration, detailing the mechanistic steps and possible products, based on output of subtask 4."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking and answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4, answer4], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, enumerating pathways from D to E, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    cot_instruction_5b = "Sub-task 5b: Predict the IR spectral features, especially the carbonyl stretching frequencies, and other relevant spectral or chemical properties for each candidate structure of compound E derived from the pathways enumerated in subtask 5a."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking and answer of subtask 5a"],
        "agent_collaboration": "CoT"
    }
    thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking5a, answer5a], cot_instruction_5b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5b.id}, predicting spectral features for candidate Es, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    debate_instruction_5c = "Sub-task 5c: Compare the predicted spectral data and mechanistic plausibility of each candidate compound E with the experimental IR peak at ~1715 cm^-1 and other contextual clues, using a combined debate and self-consistency chain-of-thought approach with scoring or weighting to evaluate and rank the candidates, based on outputs from subtasks 5a and 5b."
    debate_agents_5c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5c = self.max_round
    all_thinking5c = [[] for _ in range(N_max_5c)]
    all_answer5c = [[] for _ in range(N_max_5c)]
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": debate_instruction_5c,
        "context": ["user query", "thinking and answer of subtasks 5a and 5b"],
        "agent_collaboration": "Debate + SC_CoT"
    }
    for r in range(N_max_5c):
        for i, agent in enumerate(debate_agents_5c):
            if r == 0:
                thinking5c, answer5c = await agent([taskInfo, thinking5a, answer5a, thinking5b, answer5b], debate_instruction_5c, r, is_sub_task=True)
            else:
                input_infos_5c = [taskInfo, thinking5a, answer5a, thinking5b, answer5b] + all_thinking5c[r-1] + all_answer5c[r-1]
                thinking5c, answer5c = await agent(input_infos_5c, debate_instruction_5c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing spectral data and mechanistic plausibility for compound E candidates, thinking: {thinking5c.content}; answer: {answer5c.content}")
            all_thinking5c[r].append(thinking5c)
            all_answer5c[r].append(answer5c)
    final_decision_agent_5c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5c, answer5c = await final_decision_agent_5c([taskInfo] + all_thinking5c[-1] + all_answer5c[-1], "Sub-task 5c: Rank and evaluate candidate structures of compound E integrating spectral and mechanistic evidence.", is_sub_task=True)
    agents.append(f"Final Decision agent, ranking compound E candidates, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {
        "thinking": thinking5c,
        "answer": answer5c
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    cot_reflect_instruction_5d = "Sub-task 5d: Select the final structure of compound E from the multiple-choice options based on integrated mechanistic reasoning, spectral data comparison, and critical evaluation of alternative pathways, providing a detailed justification for the choice, based on output of subtask 5c."
    cot_agent_5d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5d = self.max_round
    cot_inputs_5d = [taskInfo, thinking5c, answer5c]
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_reflect_instruction_5d,
        "context": ["user query", "thinking and answer of subtask 5c"],
        "agent_collaboration": "Reflexion"
    }
    thinking5d, answer5d = await cot_agent_5d(cot_inputs_5d, cot_reflect_instruction_5d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5d.id}, selecting final compound E, thinking: {thinking5d.content}; answer: {answer5d.content}")
    for i in range(N_max_5d):
        feedback, correct = await critic_agent_5d([taskInfo, thinking5d, answer5d], "please review the final selection of compound E and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5d.id}, feedback on final compound E selection, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5d.extend([thinking5d, answer5d, feedback])
        thinking5d, answer5d = await cot_agent_5d(cot_inputs_5d, cot_reflect_instruction_5d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5d.id}, refining final compound E selection, thinking: {thinking5d.content}; answer: {answer5d.content}")
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {
        "thinking": thinking5d,
        "answer": answer5d
    }
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])
    cot_reflect_instruction_6 = "Sub-task 6: Perform a final validation of all intermediate assignments (compounds A through D) and the final identification of compound E to ensure consistency with reaction mechanisms, spectral data, and the multiple-choice options, correcting any discrepancies before finalizing the answer."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5d, answer5d]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking and answer of subtasks 1-5d"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, validating all compounds and final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please review the overall consistency and correctness of all compound identifications and final answer.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback on overall validation, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining overall validation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs