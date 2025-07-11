async def forward_170(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: List the substituent group R (–CH3, –COOC2H5, –Cl, –NO2, –C2H5, –COOH) attached to the benzene ring for each of the six substances (1–6)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identification of substituents, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: For each substituent R from Sub-task 1, classify it as electron-donating (activating) or electron-withdrawing (deactivating), and determine its directing effect (ortho/para vs meta)."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query","thinking of subtask 1","answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, classification and directing effect thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 3: Gather typical empirical ortho:meta:para (O:M:P) percentage distributions for electrophilic bromination for each substituent class identified in Sub-task 2, using standard literature values or reasonable approximations."
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction, "context": ["user query","thinking of subtask 2","answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, gathering O:M:P distributions, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_answers.append(answer3_i.content)
        thinkingmapping[answer3_i.content] = thinking3_i
        answermapping[answer3_i.content] = answer3_i
    answer3_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[answer3_content]
    answer3 = answermapping[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task 4: Verify that for each substituent the reported O, M, and P percentages sum to 100%, and if any inconsistencies are found, correct them."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking3, answer3]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_reflect_instruction, "context": ["user query","thinking of subtask 3","answer of subtask 3"], "agent_collaboration": "Reflexion"}
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, verifying distributions, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], "Sub-task 4: Review the percentage sums and indicate if they correctly sum to 100%.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, corrected distributions, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 5: Calculate the molecular weights of the ortho-bromo and para-bromo isomers (C6H4Br–R) for each substituent R listed in Sub-task 1."
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction, "context": ["user query","thinking of subtask 1","answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking5_i, answer5_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, MW calculation thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_answers.append(answer5_i.content)
        thinkingmapping[answer5_i.content] = thinking5_i
        answermapping[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinkingmapping[answer5_content]
    answer5 = answermapping[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    debate_instruction = "Sub-task 6a: Using the verified O:M:P percentages from Sub-task 4 and the molecular weights from Sub-task 5, compute the weight fraction of the para-bromo isomer for each substituent as weight_para/(O*mW_ortho + M*mW_meta + P*mW_para)."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking6 = [[] for _ in range(N_max)]
    all_answer6 = [[] for _ in range(N_max)]
    subtask_desc6a = {"subtask_id": "subtask_6a", "instruction": debate_instruction, "context": ["user query","thinking of subtask 4","answer of subtask 4","thinking of subtask 5","answer of subtask 5"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for agent in debate_agents:
            if r == 0:
                thinking6_i, answer6_i = await agent([taskInfo, thinking4, answer4, thinking5, answer5], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6_i, answer6_i = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, weight fraction thinking: {thinking6_i.content}; answer: {answer6_i.content}")
            all_thinking6[r].append(thinking6_i)
            all_answer6[r].append(answer6_i)
    final_decision_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6a, answer6a = await final_decision_agent([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6a: Make final decision on weight fraction calculation using full O:M:P data.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent.id}, weight fraction final thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a["response"] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking6a, answer6a]
    subtask_desc6b = {"subtask_id": "subtask_6b", "instruction": "Sub-task 6b: Review the calculations from Sub-task 6a to ensure that the full O:M:P data were used correctly in the weight-fraction formula and reconcile any discrepancies.", "context": ["user query","thinking of subtask 6a","answer of subtask 6a"], "agent_collaboration": "Reflexion"}
    thinking6b, answer6b = await cot_agent(cot_inputs, subtask_desc6b["instruction"], 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, initial critic evaluation thinking: {thinking6b.content}; answer: {answer6b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking6b, answer6b], "Sub-task 6b: Does the calculation correctly incorporate O, M, and P in the denominator and numerator? Provide True if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs.extend([thinking6b, answer6b, feedback])
        thinking6b, answer6b = await cot_agent(cot_inputs, subtask_desc6b["instruction"], i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refined critic evaluation thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b["response"] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    cot_instruction = "Sub-task 7: Sort the six substances in order of increasing para-isomer weight fraction based on the validated results from Sub-task 6b."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_instruction, "context": ["user query","thinking of subtask 6b","answer of subtask 6b"], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent([taskInfo, thinking6b, answer6b], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, sorting thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_instruction = "Sub-task 8: Compare the sorted sequence from Sub-task 7 to the provided multiple-choice options (A–D) and select the matching answer letter."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_instruction, "context": ["user query","thinking of subtask 7","answer of subtask 7"], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent([taskInfo, thinking7, answer7], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, matching answer thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8["response"] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs