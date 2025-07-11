async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Extract and record the system parameters: c(Co), [SCN-], and stability constants."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracted system parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction = "Sub-task 2: Identify all possible Co(II)-thiocyanato species and their stoichiometries."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction, "context": ["user query", "Sub-task 1 response"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identified species, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction = "Sub-task 3: Write equilibrium expressions for each Co–SCN complex using stability constants."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction, "context": ["user query", "response of subtask_2"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, wrote equilibrium expressions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction = "Sub-task 4: Formulate the cobalt mass-balance equation including all species."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction, "context": ["user query", "response of subtask_3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent([taskInfo, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulated mass-balance, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 5: Solve for free [Co2+] and [SCN-] using the mass-balance and equilibrium expressions."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask_4", "answer of subtask_4"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking5_i, answer5_i = await cot_agents[i]([taskInfo, thinking4, answer4], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, solving equilibrium, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_answers.append(answer5_i.content)
        thinkingmapping[answer5_i.content] = thinking5_i
        answermapping[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinkingmapping[answer5_content]
    answer5 = answermapping[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction = "Sub-task 6: Calculate [Co(SCN)2] using the equilibrium expression and solved free-ion concentrations."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_instruction, "context": ["user query", "response of subtask_5"], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot_agent([taskInfo, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculated [Co(SCN)2], thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction = "Sub-task 7: Compute the percentage of Co(SCN)2 and select the corresponding choice (A–D)."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking7 = [[] for _ in range(N_max)]
    all_answer7 = [[] for _ in range(N_max)]
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": debate_instruction, "context": ["user query", "thinking of subtask_6", "answer of subtask_6"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking7_i, answer7_i = await agent([taskInfo, thinking6, answer6], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7_i, answer7_i = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing percentage, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
            all_thinking7[r].append(thinking7_i)
            all_answer7[r].append(answer7_i)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on percentage and choice.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent.id}, selected final percentage, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs