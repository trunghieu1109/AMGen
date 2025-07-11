async def forward_190(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Identify and extract all functional groups and substituents in 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cyclohexan-1-one."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying functional groups, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Classify each reagent and condition (NaH/BnBr, p-toluenesulfonyl hydrazide/HCl, n-BuLi/NH4Cl, Pd/C H2) by reaction type."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, classifying reagents, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Predict the structure of product 1 after O-benzylation of the hydroxymethyl group with NaH and benzyl bromide."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query", "answer of subtask 2"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, predicting product 1, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Predict the structure of product 2 obtained by converting the ketone of product 1 into a p-toluenesulfonyl hydrazone under HCl catalysis."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "answer of subtask 3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, predicting product 2, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_reflect_instruction5 = "Sub-task 5: Predict the structure of product 3 from the Shapiro reaction of product 2 with n-BuLi and NH4Cl."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction5, "context": ["user query", "answer of subtask 4"], "agent_collaboration": "Reflexion"}
    inputs5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent5(inputs5, cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, initial Shapiro prediction, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], "Review the Shapiro reaction prediction and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == "True":
            break
        inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(inputs5, cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refined Shapiro prediction, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_sc_instruction6 = "Sub-task 6: Predict the structure of product 4 after catalytic hydrogenation (Pd/C, H2), including hydrogenolysis of the benzyl ether and reduction of all C=C bonds."
    N2 = self.max_sc
    sc_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible6 = []
    thinkmap6 = {}
    ansmap6 = {}
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_sc_instruction6, "context": ["user query", "answer of subtask 5"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        t6, a6 = await sc_agents6[i]([taskInfo, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents6[i].id}, hydrogenation prediction, thinking: {t6.content}; answer: {a6.content}")
        possible6.append(a6.content)
        thinkmap6[a6.content] = t6
        ansmap6[a6.content] = a6
    ans6_content = Counter(possible6).most_common(1)[0][0]
    thinking6 = thinkmap6[ans6_content]
    answer6 = ansmap6[ans6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction7 = "Sub-task 7: Compare the predicted structure of product 4 to the four multiple-choice options and select the matching IUPAC name."
    debate_agents7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds = self.max_round
    all_t7 = [[] for _ in range(rounds)]
    all_a7 = [[] for _ in range(rounds)]
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": debate_instruction7, "context": ["user query", "answer of subtask 6"], "agent_collaboration": "Debate"}
    for r in range(rounds):
        for i, agent in enumerate(debate_agents7):
            if r == 0:
                t7, a7 = await agent([taskInfo, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking6, answer6] + all_t7[r-1] + all_a7[r-1]
                t7, a7 = await agent(inputs7, debate_instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting IUPAC name, thinking: {t7.content}; answer: {a7.content}")
            all_t7[r].append(t7)
            all_a7[r].append(a7)
    final_decision7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision7([taskInfo] + all_t7[-1] + all_a7[-1], "Sub-task 7: Make final decision on the correct IUPAC name for product 4.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding IUPAC name, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs