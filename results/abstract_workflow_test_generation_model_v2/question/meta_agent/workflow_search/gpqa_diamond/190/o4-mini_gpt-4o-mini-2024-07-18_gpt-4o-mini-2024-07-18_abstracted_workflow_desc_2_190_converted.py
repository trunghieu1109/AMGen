async def forward_190(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Parse the query to extract the structure of 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cyclohexan-1-one and list all reagents, catalysts, and conditions for each step."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parsing query, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Classify functional groups in the starting material (ketone, alcohol, alkene) and categorize each reagent by role (base, alkylating agent, hydrazide, strong base, catalyst)."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, classifying functional groups, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction = "Sub-task 3a: Determine whether benzylation occurs at oxygen (O-alkylation) or carbon (C-alkylation), with mechanistic justification."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {"subtask_id": "subtask_3a", "instruction": cot_instruction, "context": ["thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "CoT"}
    thinking3a, answer3a = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, determining benzylation site, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a["response"] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_instruction = "Sub-task 3b: Specify the structure of product 1 using SMILES or clear description for the benzyl ether."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {"subtask_id": "subtask_3b", "instruction": cot_instruction, "context": ["thinking of subtask 3a", "answer of subtask 3a"], "agent_collaboration": "CoT"}
    thinking3b, answer3b = await cot_agent([taskInfo, thinking3a, answer3a], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, specifying product 1 structure, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b["response"] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task 4: Write the stepwise mechanism for p-toluenesulfonyl hydrazone formation from product 1 and describe product 2."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_reflect_instruction, "context": ["thinking of subtask 3b", "answer of subtask 3b"], "agent_collaboration": "Reflexion"}
    thinking4, answer4 = await cot_agent([taskInfo, thinking3b, answer3b], cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, initial hydrazone mechanism, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], "Please review the hydrazone formation mechanism and product structure for accuracy.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        thinking4, answer4 = await cot_agent([taskInfo, thinking3b, answer3b, thinking4, answer4, feedback], cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining hydrazone mechanism, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task 5a: Outline the detailed Shapiro reaction mechanism for hydrazone 2 with 2 equiv n-BuLi, highlighting deprotonation, N2 elimination, and alkene formation."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {"subtask_id": "subtask_5a", "instruction": cot_reflect_instruction, "context": ["thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "Reflexion"}
    thinking5a, answer5a = await cot_agent([taskInfo, thinking4, answer4], cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, initial Shapiro mechanism, thinking: {thinking5a.content}; answer: {answer5a.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking5a, answer5a], "Review the Shapiro mechanism steps for completeness and correct elimination.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        thinking5a, answer5a = await cot_agent([taskInfo, thinking4, answer4, thinking5a, answer5a, feedback], cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining Shapiro mechanism, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a["response"] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 5b: Predict the structure of product 3 (alkene) from the Shapiro reaction, with one-sentence mechanistic justification."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc5b = {"subtask_id": "subtask_5b", "instruction": cot_sc_instruction, "context": ["thinking of subtask 5a", "answer of subtask 5a"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking5b, answer5b = await cot_agents[i]([taskInfo, thinking5a, answer5a], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, predicting product 3 structure, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers.append(answer5b.content)
        thinkingmapping[answer5b.content] = thinking5b
        answermapping[answer5b.content] = answer5b
    answer5b_content = Counter(possible_answers).most_common(1)[0][0]
    thinking5b = thinkingmapping[answer5b_content]
    answer5b = answermapping[answer5b_content]
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b["response"] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 6a: Identify which functional groups in product 3 are susceptible to Pd/C-catalyzed hydrogenation."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc6a = {"subtask_id": "subtask_6a", "instruction": cot_sc_instruction, "context": ["thinking of subtask 5b", "answer of subtask 5b"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking6a, answer6a = await cot_agents[i]([taskInfo, thinking5b, answer5b], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identifying hydrogenation sites, thinking: {thinking6a.content}; answer: {answer6a.content}")
        possible_answers.append(answer6a.content)
        thinkingmapping[answer6a.content] = thinking6a
        answermapping[answer6a.content] = answer6a
    answer6a_content = Counter(possible_answers).most_common(1)[0][0]
    thinking6a = thinkingmapping[answer6a_content]
    answer6a = answermapping[answer6a_content]
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a["response"] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    cot_instruction = "Sub-task 6b: Describe the structure of product 4 after hydrogenation, specifying bond changes."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6b = {"subtask_id": "subtask_6b", "instruction": cot_instruction, "context": ["thinking of subtask 6a", "answer of subtask 6a"], "agent_collaboration": "CoT"}
    thinking6b, answer6b = await cot_agent([taskInfo, thinking6a, answer6a], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, describing product 4 structure, thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b["response"] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    debate_instruction = "Sub-task 7: Compare the predicted structure of product 4 with options A-D and select the correct letter."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = [[] for _ in range(self.max_round)]
    all_answer7 = [[] for _ in range(self.max_round)]
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": debate_instruction, "context": ["thinking of subtask 6b", "answer of subtask 6b"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6b, answer6b], debate_instruction, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking6b, answer6b] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(inputs7, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct option (A, B, C, or D).", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final option, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs