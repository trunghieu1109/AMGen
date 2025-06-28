async def forward_48(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the general mechanism and characteristics of sigmatropic rearrangements, focusing on the migration of terminal pi bonds into sigma bonds, and understand the thermodynamic favorability and orbital symmetry rules governing these reactions, with detailed mechanistic context."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing sigmatropic rearrangements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction_2 = "Sub-task 2: Review the specific features, mechanisms, and stereochemical outcomes of Cope and Claisen rearrangements as prototypical sigmatropic rearrangements, including detailed atom numbering and mechanistic precedents to establish a reliable framework for product prediction, based on Sub-task 1 outputs."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, reviewing Cope and Claisen rearrangements, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_instruction_3a = "Sub-task 3a: For the first reaction (1,1-dimethoxyethan-1-amine + but-3-en-2-ol + H+ + Heat), perform detailed mechanistic mapping including atom connectivity, intermediates, and curved-arrow mechanisms considering acid-catalyzed pathways, Pinner–Mannich or substitution mechanisms, and alternative routes beyond sigmatropic rearrangement."
    cot_instruction_3b = "Sub-task 3b: Generate multiple plausible mechanistic pathways for the first reaction and predict the product A for each pathway with detailed stereochemical analysis."
    cot_instruction_3c = "Sub-task 3c: Evaluate the plausibility of each proposed mechanism for the first reaction using orbital symmetry rules, literature precedents, and substrate-specific chemistry to select the most consistent product A."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    debate_agents_3c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, mechanistic mapping for first reaction, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "SC_CoT"
    }
    N3b = self.max_sc
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    for i in range(N3b):
        thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_3b.id}, generating multiple pathways for first reaction, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    answer3b_content = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[answer3b_content]
    answer3b = answermapping_3b[answer3b_content]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    N_max_3c = self.max_round
    all_thinking3c = [[] for _ in range(N_max_3c)]
    all_answer3c = [[] for _ in range(N_max_3c)]
    for r in range(N_max_3c):
        for i, agent in enumerate(debate_agents_3c):
            if r == 0:
                thinking3c, answer3c = await agent([taskInfo, thinking3b, answer3b], cot_instruction_3c, r, is_sub_task=True)
            else:
                input_infos_3c = [taskInfo, thinking3b, answer3b] + all_thinking3c[r-1] + all_answer3c[r-1]
                thinking3c, answer3c = await agent(input_infos_3c, cot_instruction_3c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating mechanisms for first reaction, thinking: {thinking3c.content}; answer: {answer3c.content}")
            all_thinking3c[r].append(thinking3c)
            all_answer3c[r].append(answer3c)
    final_decision_agent_3c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3c, answer3c = await final_decision_agent_3c([taskInfo] + all_thinking3c[-1] + all_answer3c[-1], "Sub-task 3c: Select the most plausible product A for the first reaction based on debate outputs.", is_sub_task=True)
    agents.append(f"Final Decision agent for first reaction product, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    cot_instruction_4a = "Sub-task 4a: For the second reaction ((3R,4S)-3,4-dimethylhexa-1,5-diyne + Heat), conduct a comprehensive mechanistic exploration including possible Bergman cyclization, radical pathways, and thermal rearrangements consistent with orbital symmetry rules; provide detailed intermediates and stereochemical considerations."
    cot_instruction_4b = "Sub-task 4b: Generate multiple plausible mechanistic pathways for the second reaction and predict product B for each pathway with detailed stereochemical analysis."
    cot_instruction_4c = "Sub-task 4c: Evaluate the plausibility of each proposed mechanism for the second reaction using orbital symmetry rules, literature precedents, and substrate-specific chemistry to select the most consistent product B."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    debate_agents_4c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking2, answer2], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, mechanistic exploration for second reaction, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "SC_CoT"
    }
    N4b = self.max_sc
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    for i in range(N4b):
        thinking4b, answer4b = await cot_agent_4b([taskInfo, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_4b.id}, generating multiple pathways for second reaction, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Debate"
    }
    N_max_4c = self.max_round
    all_thinking4c = [[] for _ in range(N_max_4c)]
    all_answer4c = [[] for _ in range(N_max_4c)]
    for r in range(N_max_4c):
        for i, agent in enumerate(debate_agents_4c):
            if r == 0:
                thinking4c, answer4c = await agent([taskInfo, thinking4b, answer4b], cot_instruction_4c, r, is_sub_task=True)
            else:
                input_infos_4c = [taskInfo, thinking4b, answer4b] + all_thinking4c[r-1] + all_answer4c[r-1]
                thinking4c, answer4c = await agent(input_infos_4c, cot_instruction_4c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating mechanisms for second reaction, thinking: {thinking4c.content}; answer: {answer4c.content}")
            all_thinking4c[r].append(thinking4c)
            all_answer4c[r].append(answer4c)
    final_decision_agent_4c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4c, answer4c = await final_decision_agent_4c([taskInfo] + all_thinking4c[-1] + all_answer4c[-1], "Sub-task 4c: Select the most plausible product B for the second reaction based on debate outputs.", is_sub_task=True)
    agents.append(f"Final Decision agent for second reaction product, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {
        "thinking": thinking4c,
        "answer": answer4c
    }
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    cot_instruction_5a = "Sub-task 5a: For the third reaction (2-((vinyloxy)methyl)but-1-ene + Heat), analyze the Claisen rearrangement mechanism with explicit atom numbering and connectivity, considering substrate-specific factors affecting rearrangement, and provide detailed mechanistic rationale and stereochemical outcome."
    cot_instruction_5b = "Sub-task 5b: Generate multiple plausible product C predictions based on the Claisen rearrangement mechanism and substrate-specific factors."
    cot_instruction_5c = "Sub-task 5c: Evaluate and select the most consistent product C prediction using mechanistic plausibility, stereochemical analysis, and literature precedents."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    debate_agents_5c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking2, answer2], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, mechanistic analysis for third reaction, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "SC_CoT"
    }
    N5b = self.max_sc
    possible_answers_5b = []
    thinkingmapping_5b = {}
    answermapping_5b = {}
    for i in range(N5b):
        thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking5a, answer5a], cot_instruction_5b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_5b.id}, generating multiple product predictions for third reaction, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers_5b.append(answer5b.content)
        thinkingmapping_5b[answer5b.content] = thinking5b
        answermapping_5b[answer5b.content] = answer5b
    answer5b_content = Counter(possible_answers_5b).most_common(1)[0][0]
    thinking5b = thinkingmapping_5b[answer5b_content]
    answer5b = answermapping_5b[answer5b_content]
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_instruction_5c,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "Debate"
    }
    N_max_5c = self.max_round
    all_thinking5c = [[] for _ in range(N_max_5c)]
    all_answer5c = [[] for _ in range(N_max_5c)]
    for r in range(N_max_5c):
        for i, agent in enumerate(debate_agents_5c):
            if r == 0:
                thinking5c, answer5c = await agent([taskInfo, thinking5b, answer5b], cot_instruction_5c, r, is_sub_task=True)
            else:
                input_infos_5c = [taskInfo, thinking5b, answer5b] + all_thinking5c[r-1] + all_answer5c[r-1]
                thinking5c, answer5c = await agent(input_infos_5c, cot_instruction_5c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating product predictions for third reaction, thinking: {thinking5c.content}; answer: {answer5c.content}")
            all_thinking5c[r].append(thinking5c)
            all_answer5c[r].append(answer5c)
    final_decision_agent_5c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5c, answer5c = await final_decision_agent_5c([taskInfo] + all_thinking5c[-1] + all_answer5c[-1], "Sub-task 5c: Select the most plausible product C for the third reaction based on debate outputs.", is_sub_task=True)
    agents.append(f"Final Decision agent for third reaction product, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {
        "thinking": thinking5c,
        "answer": answer5c
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    debate_instruction_6 = "Sub-task 6: Integrate and critically evaluate the predicted products A, B, and C from subtasks 3c, 4c, and 5c by comparing them against the given multiple-choice options; apply self-consistency checks, cross-validate mechanistic plausibility, and select the correct final answer choice with justification."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c", "thinking of subtask 4c", "answer of subtask 4c", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking3c, answer3c, thinking4c, answer4c, thinking5c, answer5c], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking3c, answer3c, thinking4c, answer4c, thinking5c, answer5c] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating and evaluating predicted products, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct answer choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs