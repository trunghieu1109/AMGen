async def forward_170(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_1 = "Sub-task 1: Classify each substituent group (–CH3, –COOC2H5, –Cl, –NO2, –C2H5, –COOH) attached to the benzene ring as electron-donating or electron-withdrawing, and determine their directing effects (ortho/para or meta), incorporating steric and electronic influences. Use a Self-Consistency Chain-of-Thought (SC CoT) approach to ensure classification accuracy by having multiple agents independently reason and compare results."
    N_sc_1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, classify substituents and directing effects, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    answer1_content = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[answer1_content]
    answer1 = answermapping_1[answer1_content]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    debate_instruction_2 = "Sub-task 2: Based on the classification of substituents from Sub-task 1, determine the expected regioselectivity of electrophilic bromination for each substituted benzene, focusing on the quantitative yield distribution of the para-isomer. Use empirical data or literature values for typical ortho/para ratios for each substituent. Employ a Debate pattern where one agent defends the initial yield estimations and another challenges them with alternative data or reasoning to refine and cross-validate the results."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, estimate para-isomer yields, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Make final decision on quantitative para-isomer yield estimations.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final para-isomer yield estimations, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3a = "Sub-task 3a: Collect and tabulate quantitative yield data (percentage or weight fraction) of para-isomer formation for each substituent from reliable literature or empirical sources. Explicitly cite numerical values to ground subsequent ranking in solid evidence."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, collect quantitative para-isomer yield data, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    debate_instruction_3b = "Sub-task 3b: Rank the substituents 1 to 6 in order of increasing para-isomer yield based on the quantitative data gathered in Sub-task 3a. Use a Debate pattern where one agent defends the ranking and another challenges it to ensure robustness and accuracy."
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
            agents.append(f"Debate agent {agent.id}, round {r}, rank substituents by para-isomer yield, thinking: {thinking3b.content}; answer: {answer3b.content}")
            all_thinking3b[r].append(thinking3b)
            all_answer3b[r].append(answer3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3b, answer3b = await final_decision_agent_3b([taskInfo] + all_thinking3b[-1] + all_answer3b[-1], "Sub-task 3b: Make final decision on ranking of substituents by para-isomer yield.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final ranking, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Compare the derived ranking order of substances by increasing para-isomer yield with the provided multiple-choice options (A, B, C, or D). Perform an automated cross-check to identify the matching alphabetical choice. If multiple options appear plausible, trigger a secondary review to resolve ambiguities."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3b, answer3b], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, compare ranking with multiple-choice options, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs