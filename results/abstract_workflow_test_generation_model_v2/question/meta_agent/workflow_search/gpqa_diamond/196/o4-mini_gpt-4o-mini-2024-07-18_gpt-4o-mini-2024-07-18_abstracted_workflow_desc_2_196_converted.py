async def forward_196(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Identify and classify functional groups in Compound X using the IR data (3400–2500 cm⁻¹, 1720 cm⁻¹, 1610 cm⁻¹, 1450 cm⁻¹)."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction_1,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, identifying functional groups, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Based on the functional groups identified, analyze the aromatic proton signals in the 1H NMR (8.0 ppm, d, 2H; 7.2 ppm, d, 2H) to determine the substitution pattern on the benzene ring."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction_2,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, analyzing aromatic pattern, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_reflect_instruction_3 = "Sub-task 3: Based on the aromatic substitution pattern, interpret the aliphatic 1H NMR signals (2.9 ppm, m, 1H; 1.7 ppm, m, 2H; 1.4 ppm, d, 3H; 0.9 ppm, t, 3H) to deduce the identity of the alkyl substituent."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_reflect_instruction_3,"context":["user query","thinking1","answer1","thinking2","answer2"],"agent_collaboration":"Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, interpreting aliphatic signals, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "please review the alkyl interpretation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs, cot_reflect_instruction_3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refinement, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3, "answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction_4 = "Sub-task 4: Integrate the functional-group assignment, aromatic pattern, and alkyl identity to propose the full structure of Compound X."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_instruction_4,"context":["user query","thinking1","answer1","thinking2","answer2","thinking3","answer3"],"agent_collaboration":"CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, proposing full structure, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4, "answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction_5 = "Sub-task 5: Recall the mechanism and outcome of reducing a benzoic acid derivative with red phosphorus and HI (conversion of –COOH to –CH₃)."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_instruction_5,"context":["user query","thinking4","answer4"],"agent_collaboration":"CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, recalling reduction mechanism, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5, "answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_sc_instruction_6 = "Sub-task 6: Apply the reduction transformation to the proposed structure of X to predict the final product structure."
    N2 = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers6 = []
    thinkingmapping6 = {}
    answermapping6 = {}
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot_sc_instruction_6,"context":["user query","thinking4","answer4","thinking5","answer5"],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking6_i, answer6_i = await cot_agents6[i]([taskInfo, thinking4, answer4, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, predicting reduction, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible_answers6.append(answer6_i.content)
        thinkingmapping6[answer6_i.content] = thinking6_i
        answermapping6[answer6_i.content] = answer6_i
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinkingmapping6[answer6_content]
    answer6 = answermapping6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking":thinking6, "answer":answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction_7 = "Sub-task 7: Compare the predicted product structure with the provided multiple-choice options and select the correct letter (A, B, C, or D)."
    debate_agents7 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {"subtask_id":"subtask_7","instruction":debate_instruction_7,"context":["user query","thinking6","answer6"],"agent_collaboration":"Debate"}
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents7):
            if r == 0:
                thinking7_i, answer7_i = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7_i, answer7_i = await agent(input_infos, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing options, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
            all_thinking7[r].append(thinking7_i)
            all_answer7[r].append(answer7_i)
    final_decision_agent7 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the product selection.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent7.id}, selecting correct option, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking":thinking7, "answer":answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs